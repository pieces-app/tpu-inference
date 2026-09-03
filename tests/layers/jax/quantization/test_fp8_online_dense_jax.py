"""M1 / issue #158 Option B: native (flax_nnx) VLLM_FP8_ONLINE_DENSE.

The torchax method (fork PR #17/#18) is unreachable from the flax lane --
the nnx loader derives quantization from the CHECKPOINT config, and a bf16
checkpoint has none, so the stock Fp8Config died with a KeyError before any
layer was built (measured on real hardware 2026-09-01). This suite guards
the native path: the dispatch decision, the fail-closed default, the
requant numerics, and the scale-sharding axis.

Numerics run on jax[cpu]; the dispatch/structure checks are AST-based so
they run (and can FAIL) without importing vllm/torchax.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
FP8 = ROOT / "tpu_inference" / "layers" / "jax" / "quantization" / "fp8.py"
DISPATCH = (ROOT / "tpu_inference" / "layers" / "jax" / "quantization" /
            "__init__.py")
PLATFORM = ROOT / "tpu_inference" / "platforms" / "tpu_platform.py"
E4M3_MAX = 448.0


def _jnp():
    return pytest.importorskip("jax.numpy", reason="numeric check needs jax")


def _quantize_tensor():
    """Load the quantization leaf BY FILE PATH so the real primitive runs on
    a CPU-only jax install without importing tpu_inference (which pulls
    vllm/torchax) -- same technique as tests/layers/vllm/
    test_fp8_online_requant.py. Testing a local mirror instead would prove
    nothing about the shipped code path."""
    import importlib.util
    _jnp()
    path = (ROOT / "tpu_inference" / "layers" / "common" / "quantization" /
            "__init__.py")
    spec = importlib.util.spec_from_file_location("_q_leaf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.quantize_tensor


def _cls(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def _code_only(text):
    """Strip comments: an explanatory comment must neither satisfy nor
    violate a structural assertion (the vacuous-marker trap, both ways)."""
    return "\n".join(l.split("#")[0] for l in text.splitlines())


def _meth(cls, name):
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


# ------------------------------------------------------------ F1 numerics
def test_requant_scale_is_amax_over_448_per_output_channel():
    """axis=0 on an [in, out] kernel => one scale per OUTPUT channel."""
    jnp = _jnp()
    import jax
    w = (jax.random.normal(jax.random.PRNGKey(0),
                           (128, 64)) * 3.0).astype(jnp.bfloat16)
    quantize_tensor = _quantize_tensor()
    w_q, w_s = quantize_tensor(jnp.float8_e4m3fn, w, axis=0)
    assert w_s.shape[
        -1] == 64, f"expected per-OUT scale (64,), got {w_s.shape}"
    amax = jnp.max(jnp.abs(w.astype(jnp.float32)), axis=0)
    assert float(jnp.max(jnp.abs(w_s.reshape(-1) - amax / E4M3_MAX))) < 1e-3


def test_requant_round_trip_within_e4m3_tolerance():
    jnp = _jnp()
    import jax
    w = (jax.random.normal(jax.random.PRNGKey(1),
                           (256, 32)) * 5.0).astype(jnp.bfloat16)
    quantize_tensor = _quantize_tensor()
    w_q, w_s = quantize_tensor(jnp.float8_e4m3fn, w, axis=0)
    rt = w_q.astype(jnp.float32) * w_s.reshape(1, -1)
    ref = w.astype(jnp.float32)
    rel = jnp.max(jnp.abs(rt - ref) / (jnp.abs(ref) + 1e-6))
    assert float(rel) < 0.15, f"e4m3 round-trip too lossy: {float(rel)}"


# ------------------------------------------------------- F2 dispatch gate
def test_dispatch_requires_the_env_flag_and_refuses_cleanly():
    src = DISPATCH.read_text()
    assert "VLLM_FP8_ONLINE_DENSE" in src, "dispatch never consults the flag"
    assert "Fp8OnlineConfig" in src
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "get_tpu_quantization_config")
    body = ast.get_source_segment(src, fn) or ""
    flag_at = body.find("VLLM_FP8_ONLINE_DENSE")
    assert flag_at != -1
    # Scope to the ONLINE branch: the function already raises
    # NotImplementedError above, for unsupported quantization methods.
    after = body[flag_at:]
    assert "NotImplementedError" in after, (
        "a bf16 checkpoint under --quantization fp8 WITHOUT the flag must "
        "refuse CLEANLY as the else of the flag check (it used to die with "
        "a KeyError deeper in Fp8Config)")
    assert "Fp8OnlineConfig" in after


def test_dispatch_branch_requires_empty_checkpoint_quant_config():
    """Never hijack a genuinely fp8-serialized checkpoint."""
    src = DISPATCH.read_text()
    tree = ast.parse(src)
    hits = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and "hg_quant_config" in (
            ast.get_source_segment(src, n.test) or "") and "FP8" in (
                ast.get_source_segment(src, n.test) or "")
    ]
    assert hits, ("the online branch must be conditioned on BOTH "
                  "quantization==FP8 and an EMPTY checkpoint quant config")


# --------------------------------------------------- F3 method discipline
def test_create_weights_is_a_noop_that_never_makes_an_empty_scale():
    src = FP8.read_text()
    cls = _cls(ast.parse(src), "Fp8OnlineLinearMethod")
    assert cls is not None
    cw = _meth(cls, "create_weights_jax")
    assert cw is not None
    body = ast.get_source_segment(src, cw) or ""
    assert "create_param(" not in body, (
        "create_weights must NOT create params -- an uninitialized "
        "weight_scale is exactly the garbage the fail-closed guard exists "
        "to prevent (PR #17's discipline)")
    assert "Unquantized" in body and "Merged" in body, (
        "merged (gate_up/qkv) layers must still merge+interleave at load")


def test_requant_uses_axis_0_and_never_shards_the_scale_on_the_input_axis():
    src = FP8.read_text()
    cls = _cls(ast.parse(src), "Fp8OnlineLinearMethod")
    pw = _meth(cls, "process_weights_after_loading")
    body = _code_only(ast.get_source_segment(src, pw) or "")
    assert "axis=0" in body, "per-OUTPUT-channel reduction is axis=0 for [in,out]"
    # The INVARIANT, not a mechanism: a per-OUTPUT scale must never be
    # sharded along the INPUT axis (the Fp8Tensorwise wart). This used to be
    # spelled `weight_sharding[1]`; that whole branch was unreachable (see
    # the two guards below) and the scale is now left replicated, which
    # satisfies the invariant trivially -- it is 1-D and tiny.
    assert "weight_sharding[0]" not in body, (
        "sharding a per-OUTPUT scale along the INPUT axis repeats the "
        "Fp8Tensorwise wart")
    assert "batch_features" in body and "NotImplementedError" in body, (
        "batched 3D weights must fail closed, not guess a scale layout")


def test_requant_does_not_move_a_tpu_resident_param_to_a_cpu_mesh():
    """MEASURED: this method could never complete a weight load.

    It opened `with cpu_mesh_context():` around a param that
    `assign_and_shard_param` has already committed to the TPU model mesh, so
    the first op (jnp.max inside quantize_tensor) raised "Received
    incompatible devices for jitted computation". The blockwise sibling is
    allowed to do this because IT pins its params to cpu_mesh in
    create_weights_jax; this method creates no params (see the test above),
    so it must not borrow that discipline.
    """
    src = FP8.read_text()
    cls = _cls(ast.parse(src), "Fp8OnlineLinearMethod")
    pw = _meth(cls, "process_weights_after_loading")
    body = _code_only(ast.get_source_segment(src, pw) or "")
    assert "cpu_mesh_context" not in body, (
        "requant must happen WHERE THE PARAM LIVES -- this is one of the two "
        "defects that kept the flax fp8 lane from ever serving a token")


def test_requant_does_not_build_a_sharding_from_the_always_none_mesh():
    """The other half of the same story: `self.linear_config.mesh` is None on
    the JAX lane, so `NamedSharding(self.linear_config.mesh, ...)` raises
    TypeError before a token is served. Sharding is derived from the SOURCE
    ARRAY instead -- the quantized weight keeps the source's shape, so the
    source's sharding applies unchanged."""
    src = FP8.read_text()
    cls = _cls(ast.parse(src), "Fp8OnlineLinearMethod")
    pw = _meth(cls, "process_weights_after_loading")
    body = _code_only(ast.get_source_segment(src, pw) or "")
    assert "NamedSharding(self.linear_config.mesh" not in body, (
        "linear_config.mesh is None on this lane; building a NamedSharding "
        "from it is a TypeError at load time")
    assert "sharding" in body, (
        "the source array's sharding must still be consulted")


def test_experts_and_router_are_not_online_quantized():
    src = FP8.read_text()
    cls = _cls(ast.parse(src), "Fp8OnlineConfig")
    assert cls is not None
    gqm = _meth(cls, "get_quant_method")
    body = ast.get_source_segment(src, gqm) or ""
    assert "UnquantizedFusedMoEMethod" in body, (
        "experts stay on the orthogonal MOE_REQUANTIZE path")
    assert "router" in body and "UnquantizedLinearMethod" in body, (
        "the router must stay bf16 -- routing quality is disproportionately "
        "sensitive and there is no HBM payoff")


def test_engagement_marker_is_logged():
    """The lane tooling greps this line as one half of the engagement proof
    (bank writes are the other half). A silent method = a vacuous panel."""
    src = FP8.read_text()
    assert "VLLM_FP8_ONLINE_DENSE=1: serving dense on-the-fly" in src


# ------------------------------------------------------------- F4 platform
def test_env_var_is_inherited_by_engine_workers():
    src = PLATFORM.read_text()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if (isinstance(n, ast.AnnAssign)
                and getattr(n.target, "id", "") == "additional_env_vars"):
            names = [
                e.value for e in n.value.elts if isinstance(e, ast.Constant)
            ]
            assert "VLLM_FP8_ONLINE_DENSE" in names, (
                "workers would not see the flag; the pod env alone is not "
                "enough (the dispatch runs in the worker)")
            return
    raise AssertionError("additional_env_vars not found")


# ------------------------------------------------ import-time definition order
def test_no_class_is_defined_before_its_base_in_the_same_module():
    """Python evaluates base classes AT DEFINITION TIME.

    This test exists because M1's first cut placed Fp8OnlineConfig(Fp8Config)
    ABOVE class Fp8Config -> NameError at import. compressed_tensors.py
    imports this module, so EVERY flax lane failed to boot (caught on real
    hardware 2026-09-01, after a green CPU gate: the other tests read source
    with AST and never import, so an import-time error is invisible to them).
    A source-reading suite needs at least one order/importability guard.
    """
    tree = ast.parse(FP8.read_text())
    order = {
        n.name: i
        for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef)
    }
    violations = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.id if isinstance(base, ast.Name) else None
            if name in order and order[name] > order[node.name]:
                violations.append(
                    f"{node.name} defined before its base {name}")
    assert not violations, (
        "class(es) defined before their base -> NameError at import: " +
        "; ".join(violations))


def test_module_body_has_no_forward_references_to_later_definitions():
    """Same failure family, wider net: any module-level `class X(Y)` whose Y
    is defined later in the file, and any module-level call to a function
    defined later, is an import-time bomb the AST tests would otherwise miss."""
    tree = ast.parse(FP8.read_text())
    defined_at = {}
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            defined_at[node.name] = i
    problems = []
    for i, node in enumerate(tree.body):
        if isinstance(node,
                      ast.Assign):  # module-level constant built from a call
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and defined_at.get(sub.func.id, -1) > i):
                    problems.append(f"module-level call to {sub.func.id} "
                                    f"before its definition")
    assert not problems, "; ".join(problems)


# --------------------------------------------- mm/vision exclusion (hardware)
def test_multimodal_projections_are_excluded_from_online_fp8():
    """Both lanes must refuse to online-quantize multimodal projections.

    MEASURED 2026-09-01: with the flag on, the 12B booted, PASSED ITS GATE,
    and then died on the first request:
        dot_general requires contracting dimensions to have the same shape,
        got (1120,) and (6912,)
    1120 = max_soft_tokens, 6912 = the Gemma-4 vision patch dim. A per-
    output-channel scale assumes the 2-D [in, out] text-stack layout; the mm
    projections are not that shape, and nothing on the LOAD path notices --
    the mismatch only meets an activation at inference. A green boot and a
    passed gate proved nothing, which is why this guard is structural.
    """
    src = FP8.read_text()
    cls = _cls(ast.parse(src), "Fp8OnlineConfig")
    body = _code_only(
        ast.get_source_segment(src, _meth(cls, "get_quant_method")) or "")
    for token in ("vision", "audio", "multi_modal", "router"):
        assert token in body, (
            f"the flax online config does not skip {token!r} layers -- "
            "quantizing a multimodal projection dies at first inference")
    assert "UnquantizedLinearMethod" in body


def test_torchax_lane_has_the_same_exclusion():
    """The torchax method is what 12B/Unified actually reaches today, so the
    exclusion must exist on BOTH lanes or the panel dies again."""
    p = (ROOT / "tpu_inference" / "layers" / "vllm" / "quantization" /
         "fp8.py")
    src = p.read_text()
    assert "_ONLINE_FP8_SKIP_SUBSTRINGS" in src, (
        "torchax lane has no skip list; it dispatched the online method to a "
        "vision projection on hardware")
    tree = ast.parse(src)
    assigns = [
        n for n in tree.body if isinstance(n, ast.Assign) and any(
            getattr(t, "id", "") == "_ONLINE_FP8_SKIP_SUBSTRINGS"
            for t in n.targets)
    ]
    assert assigns, "skip list is not a module-level constant"
    tokens = [
        e.value for e in assigns[0].value.elts if isinstance(e, ast.Constant)
    ]
    for token in ("vision", "audio_tower", "multi_modal", "router"):
        assert any(token in t for t in tokens), f"skip list misses {token!r}"
    # and the dispatch must consult it
    gq = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "get_quant_method":
            gq = n
    assert gq is not None
    body = _code_only(ast.get_source_segment(src, gq) or "")
    assert "_is_online_fp8_eligible" in body, (
        "the skip list exists but the dispatch never consults it")


def test_excluded_layers_return_unquantized_not_the_failclosed_raise():
    """An EXCLUDED layer under an active online-fp8 run must be served bf16.

    MEASURED 2026-09-01: the first version of the skip merely declined to
    return the online method, so excluded layers FELL THROUGH to the
    fail-closed `raise NotImplementedError` further down -- turning an
    inference-time crash into an ENGINE-INIT crash. The refusal exists for
    "you asked for fp8 and we cannot provide it", NOT for "we deliberately
    keep this layer in bf16". Order matters: the exclusion branch must
    return BEFORE the raise.
    """
    p = (ROOT / "tpu_inference" / "layers" / "vllm" / "quantization" /
         "fp8.py")
    src = p.read_text()
    tree = ast.parse(src)
    gq = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "get_quant_method":
            gq = n
    assert gq is not None
    seg = ast.get_source_segment(src, gq) or ""
    excl = seg.find("not _is_online_fp8_eligible")
    assert excl != -1, "no exclusion branch found"
    # the exclusion branch must RETURN an unquantized method...
    after = seg[excl:]
    ret = after.find("VllmUnquantizedLinearMethod")
    raise_at = after.find("raise NotImplementedError")
    assert ret != -1, "exclusion branch does not return an unquantized method"
    assert raise_at == -1 or ret < raise_at, (
        "the exclusion branch falls through to the fail-closed raise -- "
        "excluded layers would abort engine init instead of serving bf16")

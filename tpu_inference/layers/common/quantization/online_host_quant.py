"""Quantize an online-quant kernel on the HOST, before it is placed on the mesh.

MEASURED 2026-09-02 23:09Z (eval-12b-native-int8-w8a16, ONE v6e chip): the
native 12B online-int8 arm died at LOAD, inside quantize_tensor called from
Fp8OnlineLinearMethod.process_weights_after_loading, with

    RESOURCE_EXHAUSTED: Attempting to allocate 450.00M ... 353.97M free

450.00 MiB is the float32 temporary of ONE fused gate_up_proj kernel
(3840 x 30720 x 4 bytes). It could not be found because the chip already
held the ENTIRE bf16 checkpoint plus the int8 copies of every kernel
requanted so far:

  * every kernel reached the mesh in bf16 (assign_and_shard_param) and was
    requanted THERE, eagerly, with two float32 temporaries alive at once,
    each 2x the kernel;
  * the merged kernels (gate_up_proj) are routed around the streaming loader
    and requanted only after the WHOLE stream, so all of them sat in bf16
    until the end;
  * JaxAutoWeightsLoader.load_weights keeps `params_dict = dict(named_
    parameters())` for the whole stream, and process_weights_after_loading
    replaced `layer.weight` with a NEW Param -- so the old Param, and its
    bf16 device buffer, stayed alive in that dict. The inline "requant per
    module to avoid OOM" freed nothing.

Peak HBM was bf16 model + int8 copies + float32 temporaries: more than the
chip. The 4-chip native lane has 4x the HBM, which is why only the 1-chip
arms died.

What this module does: at model construction the quant method leaves a
REQUEST on the kernel Param's metadata (metadata, like `weight_loader` and
`_merged_shards`, because that is what survives nnx.eval_shape's split/merge
of the abstract model, and because the loader only ever sees the Param, not
the layer). assign_and_shard_param -- the one place every loader path
commits a host array to the mesh -- honours it: the host array is quantized
with quantize_tensor ITSELF (jitted, on the host device) and only (w_q, w_s)
are placed. The bf16 kernel never reaches the mesh. The scale is parked on
the Param until process_weights_after_loading adopts it into
layer.weight_scale.

Numerics: quantize_tensor is called unchanged (per-output-channel abs-max
scale in float32; round-to-nearest for integer targets, PR #45). jit vs eager
is bit-identical on CPU for every online dtype, including zero columns
(tests/layers/common/quantization/test_online_host_quant_numerics.py).

Leaf module: imports only jax and the quantization leaf, so the CPU gate can
execute it without vllm/torch.
"""
import dataclasses
from typing import Any, Callable

import jax
from jax.sharding import Mesh, NamedSharding

from tpu_inference.layers.common.quantization import quantize_tensor

# Metadata keys on the kernel Param.
HOST_QUANT_REQUEST = "_online_host_quant"
HOST_QUANT_SCALE = "_online_host_quant_scale"


@dataclasses.dataclass(frozen=True)
class HostQuantRequest:
    """What to do to the host array before it is placed.

    kernel_shape: the 2-D [in, out] layout the online method serves. A 3-D
        einsum kernel (o_proj's [N, H, D]) is flattened to it, exactly as the
        on-device requant did after placement.
    weight_spec / scale_spec: PartitionSpec entries for the placed kernel and
        its 1-D per-OUTPUT-channel scale. Both come from the method's
        linear_config.weight_sharding -- the spec the apply path assumes for
        the 2-D kernel -- so the placement matches the forward pass.
    """
    dtype: Any
    kernel_shape: tuple[int, ...]
    weight_spec: tuple
    scale_spec: tuple
    axis: int = 0


# quantize_tensor ITSELF, jitted. The host path must not be a second
# implementation of the numerics. static_argnums = (dtype, axis, block_size).
_quantize = jax.jit(quantize_tensor, static_argnums=(0, 2, 3))


def request_host_quant(param, request: HostQuantRequest) -> None:
    param.set_metadata(HOST_QUANT_REQUEST, request)


def host_quant_request(param) -> HostQuantRequest | None:
    return param.get_metadata().get(HOST_QUANT_REQUEST)


def _mesh_of(array: jax.Array) -> Mesh:
    """The mesh to compute on: the array's own.

    Inside the loader the active mesh is the TPU model mesh, and ANY op on a
    host-resident array under it -- eager, reshape, jit, even under
    jax.default_device -- raises "Received incompatible devices for jitted
    computation" (measured on jax 0.11.1; it is the same error the on-device
    requant hit when it borrowed cpu_mesh_context). Only a nested context
    over the array's own devices works, which is what cpu_mesh_context does
    for the merged-shard loaders.
    """
    sharding = getattr(array, "sharding", None)
    if isinstance(sharding, NamedSharding):
        return sharding.mesh
    return Mesh(sorted(array.devices(), key=lambda d: d.id), ("host", ))


def quantize_on_host(dtype, weight: jax.Array, axis: int = 0):
    """(w_q, w_s) = quantize_tensor(dtype, weight, axis), computed where
    `weight` lives."""
    with jax.set_mesh(_mesh_of(weight)):
        return _quantize(dtype, weight, axis, None)


def place_host_quantized(param, weight: jax.Array, *, mesh: Mesh,
                         put: Callable[[jax.Array, tuple], jax.Array]) -> bool:
    """Honour a HostQuantRequest on `param` for the host array `weight`.

    Returns False -- touching nothing -- when the Param carries no request,
    or when `weight` is already on `mesh`: then there is nothing to keep off
    the mesh and the on-device requant in process_weights_after_loading
    stays the path (pathways_dummy builds its dummies on the TPU).

    Otherwise quantizes on the host, sets the Param's value to
    put(w_q, weight_spec) and parks put(w_s, scale_spec) under
    HOST_QUANT_SCALE for adopt_host_quant_scale. `put` is the caller's
    placement primitive (shard_put): sharding stays the loader's business.
    """
    request = host_quant_request(param)
    if request is None:
        return False
    if not set(weight.devices()).isdisjoint(mesh.devices.flatten().tolist()):
        return False
    with jax.set_mesh(_mesh_of(weight)):
        w2d = weight.reshape(request.kernel_shape)
        w_q, w_s = _quantize(request.dtype, w2d, request.axis, None)
    del w2d
    param.set_value(put(w_q, request.weight_spec))
    del w_q
    param.set_metadata(HOST_QUANT_SCALE, put(w_s, request.scale_spec))
    return True


def adopt_host_quant_scale(param):
    """The parked scale, taken off the Param (the reference is cleared), or
    None when the Param was not host-quantized."""
    w_s = param.get_metadata().get(HOST_QUANT_SCALE)
    if w_s is None:
        return None
    param.set_metadata(HOST_QUANT_SCALE, None)
    return w_s

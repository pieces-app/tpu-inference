# p3: torchax View materialization fix.
# Gemma4Unified (gemma-4-12b) vision indexes an is_embed boolean mask into a
# lazy torchax View; stock getitem's tensor-indexing fallback reads
# self._elem, which View lacks -> AttributeError('View' has no '_elem') ->
# EngineDeadError on every image request (measured 2026-08-24 22:28 UTC).
# Fix: materialize Views via View.jax() (torchax view.py:371) before indexing.
import glob
import pathlib

candidates = glob.glob("/usr/local/lib/python3*/site-packages/torchax/ops/jtorch.py") \
    or glob.glob("/usr/lib/python3*/site-packages/torchax/ops/jtorch.py")
assert candidates, "torchax jtorch.py not found in image"
p = pathlib.Path(candidates[0])
s = p.read_text()
old = """  indexes = self._env.t2j_iso(indexes)
  return torchax.tensor.Tensor(self._elem[indexes], self._env)"""
new = """  indexes = self._env.t2j_iso(indexes)
  self_arr = self._elem if hasattr(self, "_elem") else self.jax()
  return torchax.tensor.Tensor(self_arr[indexes], self._env)"""
assert old in s, "torchax getitem pattern not found — version drift, re-inspect before shipping"
p.write_text(s.replace(old, new, 1))
pyc = p.parent / "__pycache__"
if pyc.exists():
    import shutil
    shutil.rmtree(pyc)
print(f"p3 torchax patch applied to {p}")

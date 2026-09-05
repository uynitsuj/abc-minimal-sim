"""Does the --obs-condition value actually change the sampled actions?

Instantiates BatchedPi0 three times (condition=1.0, 0.0, NaN), feeds the SAME
synthetic observation, and compares action chunks. Same rng seed each time, so
any difference is attributable to the condition alone.
"""
import sys

sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")
import numpy as np

from abc_minimal.eval_policy import PI0_CAMERA_KEY_MAP, PI0_PROMPT
from batched_policy import BatchedPi0

CKPT = "/scratch/ckpt/pi05_bottles_warpcfg/warpcfg05_s0/7499"
CFG = "pi05_bottles_warpcfg"

rng = np.random.default_rng(0)
B = 2
state = rng.normal(size=(B, 14)).astype(np.float32)
imgs = {n: rng.integers(0, 255, size=(B, 3, 168, 224), dtype=np.uint8)
        for n in PI0_CAMERA_KEY_MAP}

out = {}
for label, cond in [("o=1", 1.0), ("o=0", 0.0), ("null", float("nan"))]:
    p = BatchedPi0(CFG, CKPT, PI0_PROMPT, PI0_CAMERA_KEY_MAP, condition=cond)
    obs = p.make_obs(state, imgs)
    has = "condition" in obs[0]
    val = obs[0].get("condition")
    a = p.infer_batch(obs)
    out[label] = a
    print(f"{label:5s}: obs has condition={has} value={val} -> action[0,0,:3]={a[0,0,:3]}")

print(f"||o=1 - null|| = {np.abs(out['o=1'] - out['null']).max():.6e}")
print(f"||o=0 - null|| = {np.abs(out['o=0'] - out['null']).max():.6e}")
print(f"||o=1 - o=0||  = {np.abs(out['o=1'] - out['o=0']).max():.6e}")
print("VERDICT:", "condition PATH WORKS" if np.abs(out["o=0"] - out["null"]).max() > 1e-6
      else "o=0 and null produce IDENTICAL actions -> path broken or emb too small to matter")

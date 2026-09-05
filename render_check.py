"""RISK-1 CHECK: does the mjwarp RENDER path honor per-world geom_dataid?

Build the 2-world heterogeneous model (different bottle/bin mesh scales),
set BOTH worlds to the SAME qpos, render both. If the render path is
world-aware, the images differ (different bottle silhouette sizes at identical
poses). If it ignores per-world dataid, the images are identical -> FAIL.
"""
import sys
sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

from batched_runner import combined_model, serial_reset_rows
from abc_minimal.eval_policy import PutBottlesSimConfig

scene = PutBottlesSimConfig()
SEEDS = [20260511, 20260543]
wp.set_device("cuda:0")

model, rows = combined_model(scene, SEEDS)
q0, _ = serial_reset_rows(scene, SEEDS)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

mw = mjw.put_model(model, batch_sizes={"geom_dataid": 2})
arr = mw.geom_dataid.numpy(); arr[:] = rows
wp.copy(mw.geom_dataid, wp.from_numpy(arr.astype(np.int32), dtype=wp.int32))
dw = mjw.put_data(model, data, nworld=2, nconmax=model.nconmax, njmax=model.njmax)

# SAME qpos in both worlds -> only geometry differs
qsame = np.tile(q0[0], (2, 1)).astype(np.float32)
wp.copy(dw.qpos, wp.from_numpy(qsame, dtype=wp.float32))
mjw.forward(mw, dw)

ctx = mjw.create_render_context(mjm=model, nworld=2, cam_res=(168, 224),
                                render_rgb=[True] * model.ncam,
                                render_depth=[False] * model.ncam,
                                use_textures=True, use_shadows=True)
mjw.refit_bvh(mw, dw, ctx)
mjw.render(mw, dw, ctx)
rgba = ctx.rgb_data.numpy().view(np.uint8).reshape(2, model.ncam, 224, 168, 4)
img0, img1 = rgba[0, 0, ..., :3].astype(int), rgba[1, 0, ..., :3].astype(int)
diff = np.abs(img0 - img1)
frac = float((diff.sum(-1) > 10).mean())
print(f"pixels differing (same qpos, different world meshes): {100*frac:.2f}%")
print(f"max channel diff: {diff.max()}")
# sanity: a world differing from ITSELF must be 0
print("self-diff sanity:", float(np.abs(img0 - img0).max()))
verdict = "PASS (render is world-aware)" if frac > 0.001 else "FAIL (render ignores per-world dataid)"
print("RENDER_CHECK:", verdict)

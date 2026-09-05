"""RISK-2 CHECK: contact buffer semantics under batching.

Question: is nconmax per-world or shared across the batch? If shared, B worlds
of piling bottles overflow a budget sized for 1 and contacts silently drop.
Method: introspect the contact buffer shapes at B=1 vs B=16, then run a
high-contact settle at B=16 and report per-world ncon vs capacity.
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
wp.set_device("cuda:0")

for B in (1, 16):
    seeds = [20260511 + k for k in range(B)]
    model, rows = combined_model(scene, seeds)
    q0, _ = serial_reset_rows(scene, seeds)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mw = mjw.put_model(model, batch_sizes={"geom_dataid": B})
    arr = mw.geom_dataid.numpy(); arr[:] = rows
    wp.copy(mw.geom_dataid, wp.from_numpy(arr.astype(np.int32), dtype=wp.int32))
    dw = mjw.put_data(model, data, nworld=B, nconmax=model.nconmax, njmax=model.njmax)
    wp.copy(dw.qpos, wp.from_numpy(q0.astype(np.float32), dtype=wp.float32))
    mjw.forward(mw, dw)
    for _ in range(120):   # settle: bottles fall + rest = heavy contact phase
        mjw.step(mw, dw)
    ncon = dw.ncon.numpy() if hasattr(dw, "ncon") else None
    print(f"B={B}: model.nconmax={model.nconmax}")
    for f in ("ncon", "nconmax", "njmax"):
        a = getattr(dw, f, None)
        if a is not None:
            try:
                v = a.numpy()
                print(f"  d.{f}: shape={v.shape} values={v[:8] if v.ndim else v}")
            except Exception:
                print(f"  d.{f}: {a}")
    c = getattr(dw, "contact", None)
    if c is not None:
        d = getattr(c, "dist", None)
        if d is not None:
            print(f"  contact.dist shape={tuple(d.shape)}   <- capacity axis")
    del mw, dw
print("CONTACT_CHECK_DONE")

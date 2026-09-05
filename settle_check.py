"""Settle-consistency: full runner at B=4 vs per-seed serial mjwarp references.
PASS = each batched world's final qpos is closest to ITS OWN serial reference,
with deviations in the chaotic-settle band (arms/bin tight, bottles loose)."""
import sys
sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

from batched_runner import combined_model, serial_reset_rows, run_batched
from abc_minimal.eval_policy import PutBottlesSimConfig, scene_xml

scene = PutBottlesSimConfig()
SEEDS = [20260511, 20260512, 20260513, 20260514]
STEPS = 240

wp.set_device("cuda:0")
res = run_batched(SEEDS, steps=STEPS, gpu=0, trace=True)
q_final = res.qpos_traces[:, -1, :].astype(np.float64)

q0_rows, _ = serial_reset_rows(scene, SEEDS)
refs = []
for i, s in enumerate(SEEDS):
    from batched_runner import world_scales
    bs, binsc = world_scales(scene, s)
    m = mujoco.MjModel.from_xml_string(scene_xml(scene, bs, binsc))
    d = mujoco.MjData(m)
    d.qpos[:] = q0_rows[i]
    mujoco.mj_forward(m, d)
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d, nworld=1, nconmax=m.nconmax, njmax=m.njmax)
    decim = scene.control_decimation
    ctrl = np.zeros((1, m.nu), np.float32)
    for _ in range(STEPS):
        wp.copy(dw.ctrl, wp.from_numpy(ctrl, dtype=wp.float32))
        for _ in range(decim):
            mjw.step(mw, dw)
    refs.append(dw.qpos.numpy()[0].astype(np.float64))
    del mw, dw

ok = True
for w in range(len(SEEDS)):
    own = float(np.max(np.abs(q_final[w] - refs[w])))
    others = [float(np.max(np.abs(q_final[w] - refs[o]))) for o in range(len(SEEDS)) if o != w]
    closest = own < min(others)
    ok &= closest
    print(f"world {w} (s{SEEDS[w]}): |own|={own:.4f}  min|other|={min(others):.4f}  own-closest={closest}")
print("SETTLE:", "PASS" if ok else "FAIL")

"""Gymnasium-style VECTOR env over the batched mjwarp bottles runner.

Design note: the harness's LIBERO path wraps one env per process and vectorizes
with SyncVectorEnv. That would throw away the entire batched-runner win (102x
physics, ~14x/GPU end to end). Instead this exposes the batch DIRECTLY as a
gym.vector-style API: every method acts on all B worlds at once, in lockstep.

Contract (matches robometer_policy_learning's pi0 wrapper keys):
    obs = {"observation/state": [B, K] f32,
           "observation/image": [B, H, W, 3] u8,      # top camera, HWC
           "observation/wrist_image": [B, H, W, 3] u8, # left camera stand-in
           "prompt": [B] list[str]}
    step(actions[B, H, nu]) -> obs, reward[B], terminated[B], truncated[B], info

Rewards:
  * sparse  : per-chunk delta of the runtime placement latch (privileged, sim GT)
  * warp    : per-chunk WARP-RM velocity over the executed window (pixels only)
  * combined: warp + sparse_scale * sparse
Reward-model plumbing is a callable: reward_fn(frames[B, T, H, W, 3]) -> [B].
That is the seam where WARP-RM / ReWiND / Robometer swap in.

Hacking probes: info always carries BOTH the RM reward and the privileged
truth (placed_ever delta, composite, tip flags), so a run where RM reward
climbs while truth stalls is visible in the logs without extra instrumentation.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")

import mujoco  # noqa: E402

from abc_minimal.eval_policy import PutBottlesSimConfig  # noqa: E402
from batched_runner import (  # noqa: E402
    BatchedEvaluator,
    combined_model,
    serial_reset_rows,
)

PROMPT = "Put the plastic bottles in the bin"


class BottlesVecEnv:
    """Batched bottles env. One instance owns B worlds on one GPU."""

    metadata = {"render_modes": []}

    def __init__(self, seeds, *, chunk_steps: int = 30, max_chunks: int = 60,
                 gpu: int = 0, scene=None, reward_fn=None, reward_mode: str = "sparse",
                 sparse_scale: float = 1.0, frame_stride: int = 5):
        import mujoco_warp as mjw
        import warp as wp

        self.mjw, self.wp = mjw, wp
        self.scene = scene or PutBottlesSimConfig()
        self.seeds = [int(s) for s in seeds]
        self.B = len(self.seeds)
        self.chunk_steps = chunk_steps
        self.max_chunks = max_chunks
        self.reward_fn = reward_fn
        self.reward_mode = reward_mode
        self.sparse_scale = sparse_scale
        self.frame_stride = frame_stride
        wp.set_device(f"cuda:{gpu}")

        self.model, self._dataid_rows = combined_model(self.scene, self.seeds)
        self._q0, self.rand_records = serial_reset_rows(self.scene, self.seeds)
        self._mjdata = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self._mjdata)

        # actuator / state maps from one serial env (same as batched_runner)
        from abc_minimal.eval_policy import PutBottlesEnv
        env = PutBottlesEnv(height=168, width=224, camera_keys=("top", "left", "right"),
                            prompt="", scene=self.scene, gpu_id=None)
        env.obs = lambda: {}
        env.reset(seed=self.seeds[0])
        self.ctrl_indices = list(env.ctrl_indices)
        self.qpos_indices = np.asarray(env.qpos_indices)
        self.gripper_idx = list(env.gripper_state_indices)
        env.close()
        self._act_scale = np.ones(len(self.ctrl_indices), np.float32)
        for i in self.gripper_idx:
            self._act_scale[i] = self.scene.gripper_ctrl_max

        self.H, self.W = 168, 224
        self._cam_ids = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, n)
            for n in ("top", "left", "right")
        }
        assert all(v >= 0 for v in self._cam_ids.values()), self._cam_ids
        self._built = False

    # ---------------------------------------------------------------- setup
    def _build_sim(self):
        mjw, wp = self.mjw, self.wp
        self.mw = mjw.put_model(self.model, batch_sizes={"geom_dataid": self.B})
        arr = self.mw.geom_dataid.numpy()
        arr[:] = self._dataid_rows
        wp.copy(self.mw.geom_dataid, wp.from_numpy(arr.astype(np.int32), dtype=wp.int32))
        self.dw = mjw.put_data(self.model, self._mjdata, nworld=self.B,
                               nconmax=self.model.nconmax, njmax=self.model.njmax)
        self.ctx = mjw.create_render_context(
            mjm=self.model, nworld=self.B, cam_res=(self.W, self.H),
            render_rgb=[True] * self.model.ncam, render_depth=[False] * self.model.ncam,
            use_textures=True, use_shadows=True)
        self._built = True

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        if not self._built:
            self._build_sim()
        wp = self.wp
        wp.copy(self.dw.qpos, wp.from_numpy(np.ascontiguousarray(self._q0, np.float32), dtype=wp.float32))
        wp.copy(self.dw.qvel, wp.from_numpy(np.zeros((self.B, self.model.nv), np.float32), dtype=wp.float32))
        self.mjw.forward(self.mw, self.dw)
        self.ev = BatchedEvaluator(self.model, self.scene, self.B)
        self.ev.prime(self._q0)
        self._chunk = 0
        self._prev_placed = np.zeros(self.B, np.int32)
        return self._obs(), {}

    def step(self, actions: np.ndarray):
        """actions: [B, chunk_steps, nu_used] in policy space."""
        wp = self.wp
        frames = []
        decim = self.scene.control_decimation
        for k in range(min(self.chunk_steps, actions.shape[1])):
            ctrl = np.zeros((self.B, self.model.nu), np.float32)
            ctrl[:, self.ctrl_indices] = actions[:, k, :len(self.ctrl_indices)] * self._act_scale
            wp.copy(self.dw.ctrl, wp.from_numpy(ctrl, dtype=wp.float32))
            for _ in range(decim):
                self.mjw.step(self.mw, self.dw)
            q = self.dw.qpos.numpy().astype(np.float64)
            self.ev.update(q)
            if self.reward_fn is not None and (k % self.frame_stride == 0):
                frames.append(self._render()["top"])
        self._chunk += 1

        placed = self.ev.placed_ever.sum(1).astype(np.int32)
        sparse = (placed - self._prev_placed).astype(np.float32)
        self._prev_placed = placed

        rm = np.zeros(self.B, np.float32)
        if self.reward_fn is not None and frames:
            rm = np.asarray(self.reward_fn(np.stack(frames, axis=1)), np.float32)

        if self.reward_mode == "sparse":
            reward = sparse
        elif self.reward_mode == "warp":
            reward = rm
        else:
            reward = rm + self.sparse_scale * sparse

        truncated = np.full(self.B, self._chunk >= self.max_chunks)
        terminated = np.zeros(self.B, bool)  # lockstep: no early termination
        info = {
            "sparse_reward": sparse,
            "rm_reward": rm,
            "placed_ever": placed,                       # privileged truth
            "composite": self.ev.comp_credit.sum(1),     # privileged truth
            "bin_tipped": (self.ev.tip_step > 0),        # privileged truth
        }
        return self._obs(), reward, terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _render(self):
        self.mjw.refit_bvh(self.mw, self.dw, self.ctx)
        self.mjw.render(self.mw, self.dw, self.ctx)
        rgba = self.ctx.rgb_data.numpy().view(np.uint8).reshape(
            self.B, self.model.ncam, self.H, self.W, 4)
        return {n: np.ascontiguousarray(rgba[:, cid, :, :, :3])
                for n, cid in self._cam_ids.items()}

    def _state_rows(self):
        q = self.dw.qpos.numpy().astype(np.float64)
        s = q[:, self.qpos_indices].astype(np.float32)
        for i in self.gripper_idx:
            s[:, i] = np.clip(s[:, i] / self.scene.gripper_ctrl_max, 0.0, 1.0)
        return s

    def _obs(self):
        imgs = self._render()
        return {
            "observation/state": self._state_rows(),
            "observation/image": imgs["top"],
            "observation/wrist_image": imgs["left"],
            "prompt": [PROMPT] * self.B,
        }

    def close(self):
        pass

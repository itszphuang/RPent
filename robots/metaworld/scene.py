# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native MetaWorld scene, calibrated cameras, and fixed-horizon TCP control."""

import mujoco
import numpy as np

from robots.metaworld.initial_state import initial_scene

CAMERAS = ("corner2", "gripperPOV")
SIZE = 1024
SERVO_GAIN = 0.15
TCP_TOLERANCE_M = 0.004
# Measured with an unloaded Sawyer gripper after 50 native steps at each end.
CLOSED_GAP_M = 0.029667591142120822
OPEN_GAP_M = 0.10034364850608656


class Scene:
    def __init__(self, task_name, index):
        self.env, self.identity = initial_scene(task_name, index)
        self.model, self.data = self.env.model, self.env.data
        self.model.cam("corner2").pos[:] = [0.75, 0.075, 0.7]
        self.model.vis.map.znear = 0.01
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = SIZE
        mujoco.mj_forward(self.model, self.data)
        self.renderer = None
        self.grip_command = -1.0
        self.success = False

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.env.close()

    def calibration(self):
        m, d = self.model, self.data
        base = d.body("base").xpos
        table = d.body("tablelink")
        forward = table.xpos - base
        forward[2] = 0.0
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        boxes = np.flatnonzero(
            (m.geom_bodyid == table.id)
            & (m.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
            & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        )
        top = max(
            d.geom_xpos[i, 2] + abs(d.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i]
            for i in boxes
        )
        return {
            "coordinate_calibration": {
                "base_xyz": base.tolist(),
                "base_rotation": np.column_stack(
                    (forward, np.cross(up, forward), up)
                ).tolist(),
                "table_top_xyz": [*table.xpos[:2], top],
                "tcp": "finger_pad_midpoint",
            }
        }

    def pads(self):
        return np.stack(
            [self.data.site(n).xpos for n in ("leftEndEffector", "rightEndEffector")]
        )

    def state(self):
        pads = self.pads()
        gap = np.linalg.norm(pads[0] - pads[1])
        steps = self.env.curr_path_length
        return {
            "tcp_world": pads.mean(axis=0).tolist(),
            "opening": float(
                np.clip((gap - CLOSED_GAP_M) / (OPEN_GAP_M - CLOSED_GAP_M), 0, 1)
            ),
            "model_env_steps": steps,
            "native_total_steps": steps,
            "init_settle_steps": 0,
            "terminated": self.success,
            "truncated": steps >= 500 and not self.success,
        }

    def advance(self, delta):
        if self.success or self.env.curr_path_length >= 500:
            return False
        _, _, terminated, _, info = self.env.step(
            np.r_[np.clip(delta, -1, 1), self.grip_command]
        )
        self.success |= bool(info["success"] or terminated)
        return True

    def move_to(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Expected a finite TCP position")
        for _ in range(100):
            delta = target - self.pads().mean(axis=0)
            if np.linalg.norm(delta) <= TCP_TOLERANCE_M or not self.advance(
                SERVO_GAIN * delta / self.env.action_scale
            ):
                break
        error = float(np.linalg.norm(target - self.pads().mean(axis=0)))
        return dict(
            status="reached" if error <= TCP_TOLERANCE_M else "partial",
            position_error_m=error,
            **self.state(),
        )

    def set_gripper(self, opening):
        if not 0 <= opening <= 1:
            raise ValueError("Expected opening in [0,1]")
        hold_tcp = self.pads().mean(axis=0)
        for _ in range(50):
            diff = opening - self.state()["opening"]
            if opening in (0, 1):
                self.grip_command = 1.0 - 2.0 * opening
            else:
                self.grip_command = float(-np.clip(diff * 5.0, -1, 1))
            if abs(diff) <= 0.05 or not self.advance(
                SERVO_GAIN
                * (hold_tcp - self.pads().mean(axis=0))
                / self.env.action_scale
            ):
                break
        state = self.state()
        reached = abs(state["opening"] - opening) <= 0.05
        return dict(
            status="reached" if reached else "partial",
            reason="reached" if reached else "blocked",
            **state,
        )

    def frame(self, camera):
        if camera not in CAMERAS:
            raise ValueError("Unknown camera")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=SIZE, width=SIZE)
        renderer = self.renderer
        renderer.update_scene(self.data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        # Optical convention: X right, Y down, Z into the image.
        cam = self.data.cam(camera)
        rotation = cam.xmat.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        if camera == "corner2":
            rgb, depth = rgb[::-1, ::-1], depth[::-1, ::-1]
            rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, cam.xpos
        f = SIZE / (2 * np.tan(np.deg2rad(self.model.cam(camera).fovy.item()) / 2))
        c = (SIZE - 1) / 2
        far = self.model.stat.extent * self.model.vis.map.zfar
        depth[depth >= far * (1 - 1e-5)] = np.nan
        return {
            "rgb": np.ascontiguousarray(rgb),
            "depth_m": np.ascontiguousarray(depth),
            "intrinsic_K": [[f, 0, c], [0, f, c], [0, 0, 1]],
            "extrinsic_cam2world": transform.tolist(),
        }

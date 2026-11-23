"""Four-degree-of-freedom MetaWorld primitives in episode coordinates."""

import numpy as np

from robots.rl.backends import ScenePrimitives
from robots.rl.contracts import CAMERAS
from robots.rl.coordinates import Coordinates
from robots.rl.tools import LiberoRLPrimitives
from rpent.tools import ToolResult


class MetaWorldPrimitives(ScenePrimitives):
    cameras = CAMERAS["metaworld"]
    _model_points = LiberoRLPrimitives._model_points

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_episode_steps = min(self._max_episode_steps, 500)
        self.coordinates = Coordinates.from_control(self.env.control("calibration"))
        self.identity = {
            k: self.env.last_obs[k]
            for k in ("task_index", "rand_vec_sha256", "init_qpos_sha256")
        }
        self._state.save("coordinates.json", self.coordinates.metadata(), step=None)

    def robot_state(self):
        state = self.env.control("state")
        self.env_steps = state["model_env_steps"]
        self._success |= state["terminated"]
        tcp = self.coordinates.points(state.pop("tcp_world"))
        return {
            **state,
            "env_steps": self.env_steps,
            "pose": {"xyz": tcp.tolist(), "orientation_rpy": [np.pi, 0.0, 0.0]},
        }

    def native_success(self):
        return self._success

    def capture(self, camera):
        frame = self.env.control("frame", camera)
        rgb, depth = frame.pop("rgb"), frame.pop("depth_m")
        K, ext = (
            np.asarray(frame["intrinsic_K"]),
            np.asarray(frame["extrinsic_cam2world"]),
        )
        rows, cols = np.indices(depth.shape, dtype=np.float32)
        rays = np.stack(
            (
                (cols - K[0, 2]) / K[0, 0],
                (rows - K[1, 2]) / K[1, 1],
                np.ones_like(rows),
            ),
            axis=-1,
        )
        world = (rays * depth[..., None]) @ ext[:3, :3].T + ext[:3, 3]
        return (
            rgb,
            world,
            {**frame, "raw_point_map_frame": "sim_world", "raw_point_map_unit": "m"},
        )

    def move_to(self, xyz):
        result = self.env.control(
            "move_to", self.coordinates.world_position(xyz).tolist()
        )
        self.env_steps = result["model_env_steps"]
        self._success |= result["terminated"]
        result.pop("tcp_world")
        return ToolResult(data=result)

    def set_gripper(self, opening):
        result = self.env.control("set_gripper", opening)
        self.env_steps = result["model_env_steps"]
        self._success |= result["terminated"]
        result.pop("tcp_world")
        return ToolResult(data=result)

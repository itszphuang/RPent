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

"""MetaWorld replay, gripper calibration, TCP control, and last-step precedence."""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from robots.rl.envs.metaworld import TASK_DESCRIPTIONS, LocalMetaWorldEnv
from robots.rl.geometry import Coordinates
from robots.rl.server import (
    BINARY_EXECUTE_MEDIA_TYPE,
    _WorkerSessionHost,
    create_app,
    unpack_execute_response,
)
from robots.rl.toolkit import RLToolkit


@pytest.fixture
def local_env():
    pytest.importorskip("metaworld")
    pytest.importorskip("mujoco")
    instance = LocalMetaWorldEnv("pick-place-v3", 49)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def scene(local_env):
    return local_env.scene


def test_initial_state_and_frame(scene):
    replay = LocalMetaWorldEnv("pick-place-v3", 49)
    try:
        assert (
            scene.identity
            == replay.identity
            == {
                "task_index": 49,
                "rand_vec_sha256": "39062a007434a0e5876dadebafbd1e06252f78462a2abcb174aa8ce167ec4641",
                "init_qpos_sha256": "dccaa738d2d526e2b1b8831dd2c6f8be4544f2cc11e5074598a3e10f09cc2703",
            }
        )
    finally:
        replay.close()
    assert scene.env.curr_path_length == 0
    c = Coordinates.from_control(scene.calibration())
    point = np.array([0.12, 0.62, 0.15])
    np.testing.assert_allclose(c.points(point), [0.62, -0.12, 0.15])
    np.testing.assert_allclose(c.world_position(c.points(point)), point)
    assert c.table_height_m == pytest.approx(0.0, abs=1e-8)


def test_empty_gripper_and_lift(scene):
    for opening in (0, 1):
        grip = scene.set_gripper(opening)
        assert grip["status"] == "reached" and abs(grip["opening"] - opening) <= 0.05
        delta = np.array([0.005, 0, 0.012])
        target = scene.pads().mean(axis=0) + delta
        move = scene.move_to(target)
        assert move["status"] == "reached" and move["position_error_m"] < 0.005
    obj = scene.data.body("obj").xpos.copy()
    assert scene.move_to(obj + [0, 0, 0.10])["status"] == "reached"
    assert scene.move_to(obj)["status"] == "reached"
    scene.set_gripper(0)
    grasp = scene.data.body("obj").xpos.copy()
    start = scene.pads().mean(axis=0)
    move = scene.move_to(start + [0, 0, 0.10])
    assert move["position_error_m"] < 0.005
    displacement = scene.data.body("obj").xpos - grasp
    np.testing.assert_allclose(displacement, [0, 0, 0.10], atol=0.005)


@pytest.mark.parametrize("success", [True, False])
def test_last_native_step(scene, monkeypatch, success):
    # Use the actual integrator and horizon; control only the verdict of step 500.
    original = scene.env.step

    def step(action):
        obs, reward, term, trunc, info = original(action)
        return obs, reward, term, trunc, {**info, "success": success}

    monkeypatch.setattr(scene.env, "step", step)
    scene.env.curr_path_length = 499
    assert scene.advance(np.zeros(3))
    state = scene.state()
    assert state["terminated"] is success and state["truncated"] is (not success)
    assert state["native_total_steps"] == 500
    assert not scene.advance(np.zeros(3))


def test_batch_stops_mid_second_item_at_500(local_env):
    scene = local_env.scene
    tk = RLToolkit(
        runtime_kwargs={"env": local_env, "sam3_client": None},
        environment="metaworld",
        record_artifacts=False,
    )
    scene.env.curr_path_length = 499
    target = (np.asarray(tk._robot_state()["pose"]["xyz"]) + [0, 0, 0.10]) * 100
    res = tk.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "set_gripper", "opening": 1},
                {"action": "move_to", "xyz": target.tolist()},
                {"action": "set_gripper", "opening": 0},
            ]
        },
    )
    assert res.data["executed_count"] == 2 and res.data["stopped_at"] == 1
    assert res.data["stop_reason"] == "truncated"
    assert scene.env.curr_path_length == 500 and scene.grip_command == -1.0


def test_custom_horizon_and_cancellation(local_env):
    def cancel():
        raise InterruptedError

    with pytest.raises(InterruptedError):
        local_env.set_gripper(0, check_cancelled=cancel)
    assert local_env.state()["model_env_steps"] == 0
    assert local_env.scene.renderer is None

    short = LocalMetaWorldEnv("pick-place-v3", 49, max_episode_steps=10)
    try:
        state = short.set_gripper(0)
        assert state["model_env_steps"] == 10 and state["truncated"]
        assert short.scene.renderer is None
    finally:
        short.close()


def test_task_descriptions():
    assert len(TASK_DESCRIPTIONS) == 50


def test_mt50_registry():
    mw = pytest.importorskip("metaworld")
    assert set(TASK_DESCRIPTIONS) == set(mw.ALL_V3_ENVIRONMENTS)


class FakeScene:
    identity = {"task_index": 49}
    last_info = {"init_state_id": 49}
    steps = 499
    success = False

    def calibration(self):
        return {
            "coordinate_calibration": {
                "base_xyz": [0, 0, 0],
                "base_rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                "table_top_xyz": [0, 0, 0],
            }
        }

    def get_task_language(self):
        return "Lift the object"

    def state(self):
        return {
            "model_env_steps": self.steps,
            "tcp_world": [0.1, 0.2, 0.3],
            "opening": 1.0,
            "terminated": self.steps == 500 and self.success,
            "truncated": self.steps == 500,
        }

    def frame(self, camera):
        return {
            "rgb": np.zeros((1024, 1024, 3), np.uint8),
            "depth_m": np.ones((1024, 1024), np.float32),
            "extrinsic_cam2world": np.eye(4).tolist(),
            "intrinsic_K": np.eye(3).tolist(),
        }

    def move_to(self, target, check_cancelled):
        check_cancelled()
        self.steps += 1
        return {**self.state(), "status": "partial"}

    def close(self):
        pass


@pytest.mark.parametrize(
    ("size", "success"), [(256, False), (512, True), (1024, False)]
)
def test_worker_episode(monkeypatch, size, success):
    scene = FakeScene()
    scene.success = success
    coords = Coordinates.from_control(scene.calibration())
    point = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(coords.points(point), [0.2, -0.1, 0.3])
    np.testing.assert_allclose(coords.world_position(coords.points(point)), point)
    assert coords.table_height_m == 0
    host = _WorkerSessionHost(
        environment="metaworld",
        sam3_endpoint=None,
        sam3_timeout_s=1,
        assets_path=None,
        libero_type=None,
        gpu=None,
    )
    monkeypatch.setattr(host, "_build_env", lambda *args, **kwargs: scene)

    class Manager:
        environment = "metaworld"

        async def execute(self, sid, name, arguments):
            return host.execute({"name": name, "arguments": arguments})

    try:
        opened = host.open({"task": {"task_name": "pick-place-v3"}, "image_size": size})
        assert opened["init_state_id"] == 49
        client = TestClient(create_app(Manager()))
        response = client.post(
            "/sessions/test/execute",
            headers={"Accept": BINARY_EXECUTE_MEDIA_TYPE},
            json={"name": "view_env", "arguments": {"camera": "all"}},
        )
        response.raise_for_status()
        meta, images = unpack_execute_response(response.content)
        assert not meta["done"] and len(images) == 2
        for image in images:
            with Image.open(io.BytesIO(image)) as png:
                assert png.size == (size, size) and png.format == "PNG"
                png.verify()
        move = {"action": "move_to", "xyz": [10, 20, 30]}
        result = host.execute(
            {"name": "execute_actions", "arguments": {"actions": [move, move]}}
        )
        assert result["done"] and result["success"] is success
        assert result["truncated"] is (not success)
        data = json.loads(result["text"])
        assert data["executed_count"] == 1
        assert data["final_state"]["terminated"] is success
        assert data["final_state"]["truncated"] is (not success)
    finally:
        host.close()

"""MetaWorld replay, gripper calibration, TCP control, and last-step precedence."""

import json
from pathlib import Path

import numpy as np
import pytest

from robots.metaworld.scene import Scene
from robots.rl.actions import PositionMoveAction
from robots.rl.contracts import EPISODE_CONTRACT_VERSION
from robots.rl.coordinates import Coordinates
from robots.rl.remote_manager import validate_request


@pytest.fixture
def scene():
    instance = Scene("pick-place-v3", 49)
    yield instance
    instance.close()


def test_initial_state_and_frame(scene):
    jobs = Path(__file__).resolve().parents[4] / "tools/rl/jobs"
    dev, formal = (
        json.loads((jobs / f"metaworld-mt50-{suffix}.json").read_text())
        for suffix in ("dev50", "eval500")
    )
    row = next(r for r in dev if r["task_name"] == "pick-place-v3")
    assert scene.identity == {k: row[k] for k in scene.identity}
    assert scene.env.curr_path_length == 0
    assert len(dev) == 50 and len(formal) == 500
    for item in dev:
        rows = [item] + [r for r in formal if r["task_name"] == item["task_name"]]
        assert {r["task_index"] for r in rows} == {49, *range(10)}
        assert len({r["rand_vec_sha256"] for r in rows}) == 11
    c = Coordinates.from_control(scene.calibration())
    point = np.array([0.12, 0.62, 0.15])
    np.testing.assert_allclose(c.points(point), [0.62, -0.12, 0.15])
    np.testing.assert_allclose(c.world_position(c.points(point)), point)
    assert c.table_height_m == pytest.approx(0.0, abs=1e-8)
    validate_request(
        dict(
            task=dict(environment="metaworld", task_name="assembly-v3", seed=49),
            gpu=0,
            max_episode_steps=500,
            episode_contract_version=EPISODE_CONTRACT_VERSION,
        ),
        {0},
    )
    with pytest.raises(ValueError):
        PositionMoveAction(action="move_to", xyz=[0, 0, 0], orientation_rpy=[0, 0, 0])


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


def test_batch_stops_mid_second_item_at_500(scene, tmp_path):
    from robots.rl.toolkit import RLToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager

    class Client:
        last_obs = scene.identity

        def control(self, method, *args):
            return getattr(scene, method)(*args)

        def get_task_language(self):
            return "Lift the object"

    tk = RLToolkit(
        environment="metaworld",
        runtime_kwargs={"env": Client(), "sam3_client": None},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        output_dir=tmp_path / "run",
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

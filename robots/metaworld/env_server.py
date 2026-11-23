"""Serve a single MetaWorld scene on the RPC dispatch thread that owns EGL."""

import json
from importlib.util import find_spec
from pathlib import Path

from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin


class MetaWorldEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    def __init__(self, task_name, index):
        self.task_name, self.index = task_name, index
        self.scene = None
        spec = find_spec("rlinf")
        root = Path(next(iter(spec.submodule_search_locations)))
        config = json.loads(
            (root / "envs/sim/metaworld/metaworld_config.json").read_text()
        )
        self.language = config["TASK_DESCRIPTIONS"][task_name]
        super().__init__()

    def _register_rpc(self):
        super()._register_rpc()
        for name in ("state", "calibration", "move_to", "set_gripper", "frame"):
            self._rpc["env." + name] = lambda *a, op=name: getattr(self.scene, op)(*a)

    def get_env_meta(self):
        return dict(task_name=self.task_name, seed=self.index)

    def get_task_language(self):
        return self.language

    def reset(self):
        from robots.metaworld.scene import Scene

        if self.scene is not None:
            self.scene.close()
        self.scene = Scene(self.task_name, self.index)
        return dict(**self.scene.identity, **self.scene.state())

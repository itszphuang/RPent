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

"""Replay an index from the official MT1 task table in a fresh native scene."""

import hashlib
import pickle

import metaworld
import numpy as np


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def initial_scene(task_name, index, benchmark=None):
    if type(index) is not int or not 0 <= index < 50:
        raise ValueError("MetaWorld task index must be in [0, 49]")
    benchmark = benchmark or metaworld.MT1(task_name, seed=0)
    task = benchmark.train_tasks[index]
    env = benchmark.train_classes[task_name]()
    try:
        env.set_task(task)
        env.seed(index)
        env.reset(seed=index)
        if env.curr_path_length != 0 or env.max_path_length != 500:
            raise RuntimeError("Unexpected MetaWorld native step budget")
        identity = {
            "task_index": index,
            "rand_vec_sha256": digest(pickle.loads(task.data)["rand_vec"]),
            "init_qpos_sha256": digest(env.data.qpos),
        }
        return env, identity
    except BaseException:
        env.close()
        raise

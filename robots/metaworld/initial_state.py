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
        identity = dict(
            task_index=index,
            rand_vec_sha256=digest(pickle.loads(task.data)["rand_vec"]),
            init_qpos_sha256=digest(env.data.qpos),
        )
        return env, identity
    except BaseException:
        env.close()
        raise

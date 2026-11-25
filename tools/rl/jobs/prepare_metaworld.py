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

"""Generate both manifests from actual independent native resets (no rendering)."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import metaworld
from metaworld.env_dict import ALL_V3_ENVIRONMENTS

from robots.metaworld.initial_state import initial_scene


def rows(name):
    benchmark = metaworld.MT1(name, seed=0)
    jobs = []
    for seed in [49, *range(10)]:
        env, identity = initial_scene(name, seed, benchmark)
        env.close()
        jobs.append(
            dict(environment="metaworld", task_name=name, seed=seed, **identity)
        )
    if len({job["rand_vec_sha256"] for job in jobs}) != len(jobs):
        raise ValueError(f"{name} has duplicate official task vectors")
    return jobs


if __name__ == "__main__":
    with ProcessPoolExecutor(4) as pool:
        tasks = list(pool.map(rows, sorted(ALL_V3_ENVIRONMENTS)))
    if len(tasks) != 50:
        raise ValueError("Expected MT50")
    for suffix, jobs in (
        ("dev50", [t[0] for t in tasks]),
        ("eval500", [r for t in tasks for r in t[1:]]),
    ):
        Path(__file__).with_name(f"metaworld-mt50-{suffix}.json").write_text(
            json.dumps(jobs, indent=2) + "\n"
        )

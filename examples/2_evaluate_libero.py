# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
sys.path.append("../LIBERO")
from pprint import pprint
import matplotlib.pyplot as plt
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero import get_libero_path
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.metric import raw_obs_to_tensor_obs

import imageio
from PIL import Image
import numpy as np
import torch
import random
from torchvision import transforms
import math
import cv2
import time
import sys
sys.path.append("./examples")
from train_utils import format_libero_obs, get_task_embeddings

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

tic = time.time()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# PICK TASK: libero_object, libero_spatial
dataset_name = "libero_object"
using_ribs = False
using_ema = True
training_steps = 30000
ribs_suffix = "-ribs" if using_ribs else ""
ema_prefix = "ema_" if using_ema else ""
desc = "openvla"

# Create a directory to store the video of the evaluation
output_directory = Path(f"outputs/eval/{dataset_name}{ribs_suffix}-{desc}/")
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
pretrained_policy_path = f"outputs/reproduce-openvla/{dataset_name}{ribs_suffix}/{ema_prefix}model-{training_steps}"
print(f"[INFO] loading {pretrained_policy_path}")
# OR a path to a local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.eval()
# print(policy)
cfg = policy.config
task_embs = get_task_embeddings(cfg, dataset_name)
policy.set_task_embeddings(task_embs.to(device))
pprint(cfg)
# exit()

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = dataset_name # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
successes = 0
num_evals_per_task = 5
parallel = False
for task_id in range(task_suite.n_tasks):
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    f_name_base = f"{ema_prefix}{dataset_name}{ribs_suffix}_steps={training_steps}_{task_description}"

    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    num_success_per_task = 0

    print("running sequentially")
    env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
    }
    print(f"env camera dims: {env_args['camera_heights']}")
    env = OffScreenRenderEnv(**env_args)
    init_state_id = 0
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    with VideoWriter(output_directory, f_name_base=f_name_base, save_video=True, single_video=True) as video_writer:
        for idx in range(num_evals_per_task):
            print(f"Test {idx+1}")
            policy.reset()
            env.seed(42)
            env.reset()

            obs = env.set_init_state(init_states[init_state_id])
            init_state_id = (init_state_id) % init_states.shape[0]
            steps = 0

            for _ in range(5):  # simulate the physics without any actions
                env.step(np.zeros(7))

            with torch.no_grad():
                while steps < 350:
                    steps += 1

                    obs = format_libero_obs(obs, task_id, cfg.resize_size, device)
                    # for k, v in obs.items():
                    #     print(f"{k} shape: {v.shape}")
                    
                    actions = policy.select_action(obs)
                    actions = actions.squeeze(0).cpu().numpy()
                    actions[-1] = np.sign(actions[-1])
                    next_obs, reward, done, info = env.step(actions)
                    video_writer.append_obs(
                        next_obs, done, idx=idx, camera_name="agentview_image"
                    )

                    # check whether succeed
                    if done:
                        num_success_per_task += 1
                        successes += 1
                        break
                    obs = next_obs
            init_state_id += 1



    success_rate_per_task = num_success_per_task / num_evals_per_task
    print(f"[info] success rate on task {task_id}, {task_description}: {success_rate_per_task}")
print(f"[info] overall success rate on {task_suite_name}: {successes / (num_evals_per_task*task_suite.n_tasks)}")
print("Done evaluating")

env.close()

toc = time.time()
print(time.strftime("%H::%M::%S", time.gmtime(toc - tic)))
exit()

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)
print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)
print(env.action_space)

# Reset the policy and environments to prepare for rollout


# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())


# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")

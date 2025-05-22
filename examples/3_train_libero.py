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

"""This script trains Diffusion Policy on the LIBERO environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

from pprint import pprint
import copy
import json
import torch
from dataclasses import asdict
import time
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import wandb
import random
import safetensors
import atexit
import gc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append("../RIBS/LIBERO")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType

sys.path.append("./examples")
from train_utils import get_task_embeddings #, eval_train_policy


def main():
    # Create a directory to store the training checkpoint.
    use_wandb = True
    wandb_run_name = "libero_object_arch=droid_ribs=True_ln=False" if use_wandb else "test"
    dataset_name = "libero_object"
    use_ribs = True
    resume = False
    resume_step = 0

    ribs_suffix = "-ribs" if use_ribs else ""
    output_directory = Path(f"outputs/train/{wandb_run_name}")
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"output path: {output_directory}")

    # Select your device
    device = torch.device("cuda")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    training_steps = 80000
    save_every = 10000
    check_ema_start_freq = 500
    def save_flag (current_step):
        if current_step % save_every == 0:
            return True
        # if current_step in [5000, 10000, 15000]:
        #     return True
        # if current_step > 15000 and current_step % save_every == 0:
        #     return True
        return False
    log_freq = 100
    ema_log_freq = 500
    started_ema = False
    start_ema_threshold = 0.03

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata(f"lerobot/{dataset_name}_image")
    features = dataset_to_policy_features(dataset_metadata.features)
    pprint(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if resume:
        print('loading trained policy to resume training')
        pretrained_policy_path = f"outputs/reproduce-openvla/{dataset_name}{ribs_suffix}/model-{resume_step}"
        pretrained_ema_path = f"outputs/reproduce-openvla/{dataset_name}{ribs_suffix}/ema_model-{resume_step}"
        print(f"[INFO] loading {pretrained_policy_path}")

        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
        cfg = policy.config
        task_embs = get_task_embeddings(cfg, dataset_name).to(device)
        policy.set_task_embeddings(task_embs)
        ema_policy = copy.deepcopy(policy)
        if os.path.exists(pretrained_ema_path):
            print('ema policy was saved. loading now...')
            safetensors.torch.load_model(ema_policy, os.path.join(pretrained_ema_path, "model.safetensors"), strict=True)
            started_ema = True
        else:
            print("no ema policy was saved yet")
    else:
        print("initializing new policy")
        # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
        # we'll just use the defaults and so no arguments other than input/output features need to be passed.
        cfg = DiffusionConfig(
            input_features=input_features, 
            output_features=output_features,
            task_embedding_format="distilbert",
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            resize_size=128,
            crop_shape=(116, 116), # set to resolution to prevent cropping
            vision_backbone="resnet50",
            pretrained_backbone_weights= None, # torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
            use_group_norm=True,
            use_layer_norm_state=True,
            use_lang_encoder=False,
            use_state_encoder=False,
            lang_hidden_dim=None,
            img_hidden_dim=None,
            state_hidden_dim=128,
            noise_scheduler_type="DDIM",
            cond_mlp_dims=(1024, 512, 512),
            down_dims=(256, 512, 1024),
            num_inference_steps=16,
            use_separate_rgb_encoder_per_camera=True,
            use_ribs=use_ribs,
            ribs_frozen=True,
            ribs_path="../src/ribs-libero_object.pth",
            ema_decay=0.99,
            scheduler_warmup_steps=1000,
        )
        # pprint(dataset_metadata.features)
        task_embs = get_task_embeddings(cfg, dataset_name).to(device)

        # We can now instantiate our policy with this config and the dataset stats.
        policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats, task_embeddings=task_embs)
        ema_policy = copy.deepcopy(policy)

    pprint(cfg)
    print(policy)
    train_param_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {train_param_count}")

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        "observation.images.image": [10*i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.images.wrist_image": [10*i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    transform = transforms.Compose(
        [
            transforms.Resize(cfg.resize_size),
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.05),
            # transforms.Normalize(
            #     mean=dataset_metadata.stats["observation.images.image"]["mean"],
            #     std=dataset_metadata.stats["observation.images.image"]["std"],
            # ),
        ]
    )

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(
        f"lerobot/{dataset_name}_image", 
        delta_timestamps=delta_timestamps,
        image_transforms=transform,
        # task_embs=task_embs,
    )
    # print(dataset[0]["observation.images.image"].shape)
    # print(dataset[0]["observation.state"].shape, dataset[0]['observation.state'])
    # exit()

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(
        policy.parameters(), 
        lr=cfg.optimizer_lr, 
        betas=cfg.optimizer_betas,
        weight_decay=cfg.optimizer_weight_decay,
    )
    if cfg.scheduler_name == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.scheduler_warmup_steps,
            num_training_steps=training_steps
        )
    elif cfg.scheduler_name == 'cosine':
        from torch.optim.lr_scheduler import LambdaLR
        def cosine_schedule_with_warmup(warmup_steps, total_steps, eta_min=1e-6, base_lr=1e-4):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return current_step / warmup_steps
                else:
                    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    decay_factor = (eta_min / base_lr) + (1 - (eta_min / base_lr)) * cosine_decay
                    return decay_factor
            return lr_lambda
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=cosine_schedule_with_warmup(
                warmup_steps=cfg.scheduler_warmup_steps,
                total_steps=training_steps,
                eta_min=cfg.optimizer_lr_min,
                base_lr=cfg.optimizer_lr
            )
        )
    else:
        raise NotImplementedError("Need a scheduler")
    
    num_workers=16
    print(f"[Info] num workers: {num_workers}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        # multiprocessing_context="spawn",
        persistent_workers=True,
        prefetch_factor=2
    )
    print(f'[INFO] number of batches {len(dataloader)}')

    # Run training loop.
    step = resume_step if resume else 0
    if resume:
        scheduler.step(resume_step)
    done = False
    losses = []
    # success_rates = []
    # val_steps = []

    # Prepare models for training
    policy.train()
    policy.to(device)
    for p in ema_policy.parameters():
        p.requires_grad=False
    ema_policy.to("cpu")
    ema_policy.eval()

    def update_ema(model, ema_model, decay):
        ema_model.to(device)
        with torch.no_grad():
            for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
        ema_model.to('cpu')

    print("starting training")
    if use_wandb:
        run = wandb.init(
            project="ribs",  # Specify your project
            name=wandb_run_name,
            config={**asdict(cfg)}        
        )

    while not done:
        print(f"refreshing dataloader at step {step}")
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # for k, v in batch.items():
            #     if type(v) != list: print(f"{k}: {v.shape}, {type(v)}")
            # exit()
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.detach().item())
            log_dict = {}

            step += 1
            if started_ema:
                update_ema(policy, ema_policy, cfg.ema_decay)
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}, lr: {scheduler.get_last_lr()}")
                log_dict["train_loss"] = loss.item()
            if step % ema_log_freq == 0 and started_ema:
                ema_policy.to(device)
                with torch.no_grad():
                    ema_loss, _ = ema_policy.forward(batch)
                    print(f"ema loss: {ema_loss.item():.3f}")
                ema_policy.to('cpu')
                log_dict["ema_loss"] = ema_loss.item()
            if save_flag(step):
                print(f"saving models at step {step}")
                policy.save_pretrained(output_directory/f"model-{step}")
                if started_ema:
                    ema_policy.save_pretrained(output_directory/f"ema_model-{step}")
            if not started_ema and step % check_ema_start_freq == 0:
                moving_avg_loss = np.mean(losses[-min(len(losses), 2000):])
                print(f'moving average loss: {moving_avg_loss}')
                if moving_avg_loss < start_ema_threshold:
                    print(f"[INFO] starting ema updates at step {step}")
                    ema_policy.to(device)
                    ema_policy.load_state_dict(policy.state_dict(), strict=True)
                    for p in ema_policy.parameters():
                        p.requires_grad = False
                    ema_policy.to("cpu")
                    started_ema = True
            if use_wandb:
                run.log(log_dict, step=step)            
            if step >= training_steps:
                done = True
                break

    print("Run ID:", run.id)
    print("Run URL:", run.url)
    print("Run name:", run.name)
    run.finish()

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory/"final_model")
    ema_policy.save_pretrained(output_directory/"final_ema_model")

    # train_stats = {
    #     'train loss' : losses,
    #     'success_rates': success_rates,
    #     'val steps' : val_steps,
    # }
    # current_time = time.strftime("%Y:%m:%d_%H:%M", time.localtime())
    # stats_name = f"{dataset_name}{ribs_suffix}_{current_time}.json"
    # with open(stats_name, "w") as f:
    #     json.dump(train_stats, f)
    # print(f'saved losses and temp success rates at {stats_name}')


def cleanup():
    print("Cleaning up GPU memory...")
    torch.cuda.empty_cache()
    print('collecting garbage')
    gc.collect()

def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}. Cleaning up before exit...")
    cleanup()
    sys.exit(0)

if __name__ == "__main__":
    import traceback
    atexit.register(cleanup)

    import signal
    # Register cleanup for Ctrl+C and termination signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination

    try:
        main()
        cleanup()
    except Exception as e:
        print("uncaught exception:", e)
        traceback.print_exc()
import torch
from torchvision import transforms
import numpy as np
import math
import sys
sys.path.append("../LIBERO")
from libero.lifelong.utils import get_task_embs
from libero.libero.benchmark import get_benchmark


def get_task_embeddings(cfg, dataset_name):
    task_suite = get_benchmark(dataset_name)(0)
    descriptions = []
    for i in range(task_suite.n_tasks):
        descriptions.append(task_suite.get_task(i).language)

    embeds = get_task_embs(cfg, descriptions)
    return embeds

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def format_libero_obs(obs, task_id, resize_size, device):
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize_size),
    ])

    item = {
        "observation.images.image": [],
        "observation.images.wrist_image": [],
        "observation.state": [],
        "task_index" : None,
    }
    def modify_item(obs_sample):
        image = img_transform(np.ascontiguousarray(obs_sample["agentview_image"][::-1,::-1]))
        item["observation.images.image"].append(image.to(device))
        wrist_image = img_transform(np.ascontiguousarray(obs_sample["robot0_eye_in_hand_image"][::-1,::-1]))
        item["observation.images.wrist_image"].append(wrist_image.to(device))
        robot_pos = np.concatenate((obs_sample['robot0_eef_pos'], quat2axisangle(obs_sample["robot0_eef_quat"]), obs_sample["robot0_gripper_qpos"]))
        item["observation.state"].append(torch.tensor(robot_pos, device=device, dtype=torch.float32))


    task_id = torch.tensor(task_id).reshape(-1)
    item["task_index"] = task_id
    if type(obs) == np.ndarray:
        for i in range(len(obs)):
            modify_item(obs[i])
    else:
        modify_item(obs)


    item["observation.images.image"] = torch.concatenate(item["observation.images.image"], dim=0).unsqueeze(0)
    item["observation.images.wrist_image"] = torch.concatenate(item["observation.images.wrist_image"], dim=0).unsqueeze(0)
    item["observation.state"] = torch.concatenate(item["observation.state"], dim=0).unsqueeze(0)
    # for k, v in item.items():
    #     print(f"{k}: {v.shape}")

    return item

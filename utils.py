import os
import cv2
import glob
import math
import json
import torch
import random
import numpy as np

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def set_random_seed(seed=42, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def load_from_json(args, json_path):
    with open(json_path, 'r') as f:
        json_obj = json.load(f)
    for k, v in json_obj.items():
        if k in args.__dict__:
            setattr(args, k, v)
        else:
            args.__dict__[k] = v
    return args

def generate_video(file_path, generate_video=None):
    if generate_video is None:
        return
    file_path += "/videos/record_" + str(generate_video)
    file_list = os.listdir(file_path)
    file_list.sort()
    fps = 10
    size = (512, 512)
    video = cv2.VideoWriter(file_path + "/video_record_{}.mp4".format(generate_video),\
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    for item in file_list:
        if item.endswith('.png'):
            item = file_path + '/' + item
            img = cv2.imread(item)
            video.write(img)
    video.release()
    for file in glob.glob(os.path.join(file_path, "*.png")):
        os.remove(file)
        
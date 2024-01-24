import os
import cv2
import gym
import math
import json
import utils
import wandb
import argparse
import datetime
import itertools
import numpy as np
import taichi as ti
import torch
from torch.nn import Upsample
torch.set_num_threads(16)
from sac import SAC
from replay_memory import ReplayMemory
import dittogym

parser = argparse.ArgumentParser(description='DittoGym Project')
parser.add_argument('--env_name', default="obstacle-fine-v0",
                    help='name of the environment to run')
parser.add_argument('--coarse_model_path', type=str, default="", metavar='G',
                    help='path of the coarse model')
parser.add_argument("--residual", type=bool, default=True,
                    help="if training residual policy (default: False)")
parser.add_argument('--residual_model_path', type=str, default="", metavar='G',
                    help='path of the residual model')
parser.add_argument("--coarse_action_res", type=int, default=8,
                    help="coarse action resolution")
parser.add_argument("--residual_action_res", type=int, default=16,
                    help="coarse action resolution")
parser.add_argument('--config_file_path', type=str, default="", metavar='G',
                    help='path of the config file')
parser.add_argument('--start_steps', type=int, default=0, metavar='N',
                    help='steps sampling random actions (default: 6000)')
parser.add_argument('--max_episode_steps', type=int, default=10000, metavar='N',
                    help='maximum steps of an episode (default: 10000)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: True)')
parser.add_argument("--cuda_deterministic", type=bool, default=False,
                    help="sets flags for determinism when using CUDA (potentially slow!) (default: False)")
parser.add_argument('--wandb', type=bool, default=True, 
                    help='if use wandb (default: True)')
parser.add_argument('--visualize', type=bool, default=True,
                    help='if save visualization results (default: True)')
parser.add_argument('--visualize_interval', type=int, default=0,
                    help='visualization interval (default: 0)')
args = parser.parse_args()

# save file path
args.name = "test_" + args.env_name + "_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
if args.config_file_path is not None:
    args = utils.load_from_json(args, args.config_file_path)
if not os.path.exists("./results"):
    os.makedirs("./results")
file_path = os.path.join(current_directory, "./results/" + args.name)
args.save_file_name = file_path
args.start_steps = 0
if not os.path.exists(file_path):
    os.makedirs(file_path)
json.dump(args.__dict__, open(file_path + "/config.json", 'w'), indent=4)
upsampled_action_res = args.action_res * args.action_res_resize
        
# Taichi
ti.init(arch=ti.gpu, random_seed=args.seed)

# GUI
if args.visualize:
    gui = ti.GUI("DittoGym", res=512, show_gui=False)

# Wandb
if args.wandb:
    wandb.init(project=args.env_name, name=args.name)
    wandb.config.update(args)
    wandb.define_metric("total_num_steps")
    wandb.define_metric("episode_num")
    wandb.define_metric("train_step_reward", step_metric="total_num_steps")
    wandb.define_metric("train_locomotion", step_metric="total_num_steps")
    wandb.define_metric("train_split", step_metric="total_num_steps")
    wandb.define_metric("train_robot_target_distance", step_metric="total_num_steps")
    wandb.define_metric("train_robot_ball_distance", step_metric="total_num_steps")
    wandb.define_metric("train_ball_target_distance", step_metric="total_num_steps")
    wandb.define_metric("train_episode_reward", step_metric="episode_num")
    wandb.define_metric("train_episode_length", step_metric="episode_num")

# Device
device = torch.device("cuda" if args.cuda else "cpu")

# Environment
env = gym.make(args.env_name, cfg_path=file_path + "/config.json", wandb_logger=wandb)

# Random
utils.set_random_seed(args.seed, args.cuda_deterministic)
env.action_space.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

if args.residual:
    agent.load_coarse_model(filename=args.coarse_model_path, action_res=args.coarse_action_res)
    agent.load_model(filename=args.residual_model_path)
    exp_upsample_list = []
    for scale_factor in [1, 2, 4, 8]:
        exp_upsample_list.append(Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True))
else:
    agent.load_model(filename=args.coarse_model_path)

# Training & Evaluation Loop
total_numsteps = 0
updates = 0
visualize_gap = 0
for i_episode in itertools.count(1):
    episode_steps = 0
    episode_reward = 0
    episode_normalize_reward = 0
    done = False
    state = env.reset()
    env.render(gui, record=False)
    if args.visualize and total_numsteps >= args.start_steps\
        and visualize_gap == args.visualize_interval:
        env.render(gui, record=True, record_id=total_numsteps)
        generate_video = total_numsteps
        visualize_gap = 0
        render = True
    else:
        generate_video = None
        render = False
        if not total_numsteps >= args.start_steps:
            visualize_gap = 0
        else:
            visualize_gap += 1
            
    # training loop
    while not done:
        if args.start_steps > total_numsteps:
            final_action = env.action_space.sample()
        else:
            if args.residual:
                coarse_action, coarse_mean, coarse_std = agent.select_coarse_action(state, task=args.name)
                coarse_action_img = coarse_action
                coarse_action = coarse_action.reshape(1, 2, args.coarse_action_res, args.coarse_action_res)
                coarse_action = exp_upsample_list[int(math.log2(args.action_res / args.coarse_action_res))]\
                    (torch.tensor(coarse_action, dtype=torch.float32, device=device)).detach().cpu().numpy()[0]
                residual_action, mask = agent.select_action(state, coarse_action, task=args.name)
                final_action = mask * residual_action + (1 - mask) * coarse_action.reshape(-1)
            else:
                final_action, _ = agent.select_action(state, task=args.name)

            # log action image (only log x direction)
            if args.wandb and total_numsteps % 200 == 0:
                final_action_ = ((final_action.reshape(2, args.action_res, args.action_res)[0] + 1) / 2 * 255)
                final_action_ = np.clip(cv2.resize(final_action_, (upsampled_action_res, upsampled_action_res),\
                    interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
                wandb.log({"final_action_x": wandb.Image(final_action_)})
    
                if args.residual:
                    coarse_action_img = exp_upsample_list[1](torch.tensor(coarse_action_img.reshape(2, args.coarse_action_res,\
                        args.coarse_action_res), dtype=torch.float32, device=device).unsqueeze(0)).detach().cpu().numpy()[0]
                    coarse_action_ = (coarse_action_img[0] + 1) / 2 * 255
                    coarse_action_ = np.clip(cv2.resize(coarse_action_, (upsampled_action_res, upsampled_action_res),\
                        interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
                    wandb.log({"coarse_action_x": wandb.Image(coarse_action_)})

                    mask_ = (mask.reshape(2, args.action_res, args.action_res)[0] * 255)
                    mask_ = np.clip(cv2.resize(mask_, (upsampled_action_res, upsampled_action_res),\
                        interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
                    wandb.log({"mask_x": wandb.Image(mask_)})

        next_state, reward, terminated, truncated, _ = env.step(final_action)
        # render
        if args.visualize and render:
            env.render(gui, record=True)
        done = truncated or terminated
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        # Ignore the "done" signal if it comes from hitting the time horizon.
        mask = 1 if truncated else float(not terminated)
        state = next_state

        if args.wandb:
            wandb.log({'total_num_steps': total_numsteps})
            wandb.log({'train_step_reward': reward})
    
    if args.wandb:
        wandb.log({'episode_num': i_episode})
        wandb.log({'train_episode_reward': episode_reward})
        wandb.log({'train_episode_length': episode_steps})
                
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"\
        .format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    utils.generate_video(file_path, generate_video)

    if total_numsteps > args.max_num_steps:
        break

env.close()

import os
import math
import torch
from torch.optim import Adam
from torch.nn import Upsample
import torch.nn.functional as F
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.num_inputs = num_inputs
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_res = args.action_res
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.exp_upsample_list = [Upsample(scale_factor=i, mode='bicubic', align_corners=True) for i in [1, 2, 4, 8]]

        # for reward normalization
        self.momentum = args.momentum
        self.mean = 0.0
        self.var = 1.0

        # critic
        self.upsampled_action_res = args.action_res * args.action_res_resize
        self.critic = QNetwork(num_inputs, self.action_res,\
            self.upsampled_action_res, args.hidden_size).to(device=self.device)
        self.critic_target = QNetwork(num_inputs, self.action_res,\
            self.upsampled_action_res, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

        # actor
        self.policy = GaussianPolicy(num_inputs, self.action_res,\
            self.upsampled_action_res, args.residual, args.coarse2fine_bias).to(device=self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        # auto alpha            
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.Tensor([action_space.shape[0]]).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
        
    def select_action(self, state, coarse_action=None, task=None):
        state = (torch.FloatTensor(state) / 255.0 * 2.0 - 1.0).to(self.device).unsqueeze(0)
        if coarse_action is not None:
            coarse_action = torch.FloatTensor(coarse_action).to(self.device).unsqueeze(0)
        if task is None or "shapematch" not in task:
            action, _, _, _, mask = self.policy.sample(state, coarse_action)
        else:
            _, _, action, _, mask = self.policy.sample(state, coarse_action)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]
        if coarse_action is not None:
            mask = mask.detach().cpu().numpy()[0]
        return action, mask

    def select_coarse_action(self, state, coarse_action=None, task=None):
        state = (torch.FloatTensor(state) / 255.0 * 2.0 - 1.0).to(self.device).unsqueeze(0)
        if coarse_action is not None:
            coarse_action = torch.FloatTensor(coarse_action).to(self.device).unsqueeze(0)
        if task is None or "shapematch" not in task:
            action, _, mean, std, _ = self.coarse_policy.sample(state, coarse_action)
            action = action.detach().cpu().numpy()[0]
            mean = mean.detach().cpu().numpy()[0]
            std = std.detach().cpu().numpy()[0]
            return action, mean, std
        else:
            _, _, action, _, _ = self.coarse_policy.sample(state, coarse_action)
            action = torch.tanh(action)
            action = action.detach().cpu().numpy()[0]
            return action, None, None

    def reward_normalization(self, rewards):
        # update mean and var for reward normalization
        batch_mean = torch.mean(rewards)
        batch_var = torch.var(rewards)
        self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
        std = torch.sqrt(self.var)
        normalized_rewards = (rewards - self.mean) / (std + 1e-8)
        return normalized_rewards

    def update_parameters(self, memory, updates):
        # sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch
        ) = memory.sample(self.args.batch_size)
        state_batch = (torch.FloatTensor(state_batch) / 255.0 * 2.0 - 1.0).to(self.device)
        next_state_batch = (torch.FloatTensor(next_state_batch) / 255.0 * 2.0 - 1.0).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        # normalize rewards
        reward_batch = self.reward_normalization(reward_batch)

        # # SAC
        # critic
        with torch.no_grad():
            if self.args.residual:
                next_original_action = self.upsample_coarse_action(next_state_batch)
                next_state_pi, next_state_log_pi, _, _, mask = self.policy.sample(next_state_batch, next_original_action)
                next_state_pi = mask * next_state_pi + (1 - mask) * next_original_action.reshape(self.args.batch_size, -1)
            else:
                next_state_pi, next_state_log_pi, _, _, _ = self.policy.sample(next_state_batch)    
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_pi)
            # only force fine policy to explore
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # two Q-functions to mitigate positive bias in the policy improvement step
        # JQ = 𝔼(st,at)~D[0.5(Q(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        # update critic
        self.critic_optim.zero_grad()
        qf_loss.backward()
        for params in self.critic.parameters():
            torch.nn.utils.clip_grad_norm_(params, max_norm=10)
        self.critic_optim.step()
        
        # actor
        if self.args.residual:
            with torch.no_grad():
                coarse_action = self.upsample_coarse_action(state_batch)
            pi, log_pi, _, std, mask = self.policy.sample(state_batch, coarse_action)
            pi = mask * pi + (1 - mask) * coarse_action.reshape(self.args.batch_size, -1)
        else:
            pi, log_pi, _, std, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss_ = ((self.alpha * log_pi) - min_qf_pi).mean()
        if self.args.residual:
            # regularize mask to be close to 0 
            mask_regularize_loss = self.args.coarse2fine_penalty *\
                torch.norm(mask.reshape(mask.shape[0], -1), dim=1).mean() / self.args.action_res
            policy_loss = policy_loss_ + mask_regularize_loss
        else:
            policy_loss = policy_loss_ 
            mask_regularize_loss = torch.zeros(1).to(self.device)
        # update policy
        self.policy_optim.zero_grad()
        policy_loss.backward()
        for params in self.policy.parameters():
            torch.nn.utils.clip_grad_norm_(params, max_norm=10)
        self.policy_optim.step()

        # update alpha (if automatic_entropy_tuning is True)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(),-torch.mean(log_pi).item(), self.alpha,\
            torch.norm(std.reshape(std.shape[0], -1), dim=1).mean().item() / (self.args.action_res**2), mask_regularize_loss.item()

    def upsample_coarse_action(self, state_batch):
        coarse_pi, _, _, _, _ = self.coarse_policy.sample(state_batch)
        coarse_pi = coarse_pi.reshape(self.args.batch_size, 2,\
            self.args.coarse_action_res, self.args.coarse_action_res)
        return self.exp_upsample_list[int(math.log2(self.args.action_res / self.args.coarse_action_res))](coarse_pi)

    # save model parameters
    def save_model(self, filename):
        checkpoint = {
            "mean": self.mean,
            "var": self.var,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
        }
        torch.save(checkpoint, filename + ".pth")

    # load model parameters
    def load_model(self, filename, for_train=False):
        print('Loading models from {}...'.format(filename))
        checkpoint = torch.load(filename)
        mean = checkpoint.get("mean")
        var = checkpoint.get("var")
        if mean is not None:
            self.mean = mean
        if var is not None:
            self.var = var
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        if for_train:
            self.policy_optim.load_state_dict(checkpoint["policy_optim"])
            self.critic_optim.load_state_dict(checkpoint["critic_optim"])
    
    # load coarse model
    def load_coarse_model(self, filename, action_res):
        print('Loading coarse models from {}...'.format(filename))
        self.coarse_policy = GaussianPolicy(self.num_inputs, action_res,\
            self.upsampled_action_res, False, self.args.coarse2fine_bias).to(self.device)
        checkpoint = torch.load(filename)
        self.coarse_policy.load_state_dict(checkpoint["policy"])
        

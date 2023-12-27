import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

EPSILON = 1e-6
SOFT_PLUS_K = 3
SOFT_PLUS_B = -1
MAX_STD = 6
MIN_STD = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Network Weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# CNN Encoder
class ObservationEncoder(nn.Module):
    def __init__(self, num_inputs, dim, channel=32):
        super(ObservationEncoder, self).__init__()
        self.channel = channel
        self.conv0 = nn.Conv2d(num_inputs + 2, channel, kernel_size=8, stride=4)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1)
        self.apply(weights_init_)

        self.position_encoder = torch.zeros(2, dim, dim).to(device)
        for i in range(dim):
            for j in range(dim):
                self.position_encoder[:, i, j] = torch.tensor([(i - int(dim / 2)) / int(dim / 2),\
                    (j - int(dim / 2)) / int(dim / 2)]).to(device)
        self.position_encoder = self.position_encoder.reshape(1, 2, dim, dim)

    def forward(self, x):
        x = torch.cat([x, self.position_encoder.repeat(x.shape[0], 1, 1, 1)], axis=1)
        x = F.elu(self.conv0(x))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        return x


# Critic Network
class QNetwork(nn.Module):
    def __init__(self, num_inputs, action_res, max_act_res, hidden_dim):
        super(QNetwork, self).__init__()
        self.action_res = action_res
        self.interpolate_rate = 64 / action_res
        self.interpolate = nn.Upsample(scale_factor=self.interpolate_rate, \
            mode='bicubic', align_corners=True)
        
        if max_act_res == 64: # obs and act have same resolution, can be merged
            self.obs_1 = ObservationEncoder(num_inputs + 2, max_act_res, 32)
            self.obs_2 = ObservationEncoder(num_inputs + 2, max_act_res, 32)
            self.merge = True
        else:
            self.obs_1 = ObservationEncoder(num_inputs, 64, 16)
            self.obs_2 = ObservationEncoder(num_inputs, 64, 16)
            self.act_1 = ObservationEncoder(2, 64, 16)
            self.act_2 = ObservationEncoder(2, 64, 16)
            self.merge = False
            
        # Q1 architecture
        self.linear1 = nn.Linear(32 * 4 * 4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.linear4 = nn.Linear(32 * 4 * 4, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        action = self.interpolate(action.reshape(-1, 2, self.action_res, self.action_res))
        if self.merge:
            s_a = torch.cat([state, action], axis=1)
            feature_1 = self.obs_1(s_a).reshape(-1, 32 * 4 * 4)
            feature_2 = self.obs_2(s_a).reshape(-1, 32 * 4 * 4)
        else:
            state_feature_1 = self.obs_1(state).reshape(-1, 16 * 4 * 4)
            state_feature_2 = self.obs_2(state).reshape(-1, 16 * 4 * 4) 
            action_feature_1 = self.act_1(action).reshape(-1, 16 * 4 * 4)
            action_feature_2 = self.act_2(action).reshape(-1, 16 * 4 * 4)
            feature_1 = torch.cat([state_feature_1, action_feature_1], dim=1)
            feature_2 = torch.cat([state_feature_2, action_feature_2], dim=1)

        x1 = F.elu(self.linear1(feature_1))
        x1 = F.elu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.elu(self.linear4(feature_2))
        x2 = F.elu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


# Actor Network
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, action_res, max_action_res, residual=False, bias=0):
        super(GaussianPolicy, self).__init__()
        if max_action_res == 64:
            self.merge = True
        else:
            self.merge = False
        if residual:
            if self.merge:
                self.obs = ObservationEncoder(num_inputs + 2, 64, 32)
            else:
                self.obs = ObservationEncoder(num_inputs, 64, 16)
                self.obs_a = ObservationEncoder(2, 64, 16)
            self.upsample = nn.Upsample(scale_factor=64 / action_res,\
                mode='bicubic', align_corners=True)
            self.residual = True
        else:
            self.residual = False
            self.obs = ObservationEncoder(num_inputs, 64, 32)
        self.bias = bias
        self.action_res = action_res
        if action_res == 4: # res: 4->8->8->4
            self.mean = nn.Sequential(
                nn.Linear(32 * 4 * 4, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 32))
            self.std = nn.Sequential(
                nn.Linear(32 * 4 * 4, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 32))
        elif action_res == 8: # res: 4->8->8->8
            self.mean = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1)
            )
            self.std = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1)
            )
        elif action_res == 16: # res: 4->8->16->16
            self.mean = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1)
            )
            self.std = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1)
            )
            if self.residual: # generate additional mask
                self.mask = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1)
                )
        elif action_res == 32: # res: 4->8->16->32
            self.mean = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, padding=1)
            )
            self.std = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, padding=1)
            )
            if self.residual: # generate additional mask
                self.mask = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, padding=1)
                )
        elif action_res == 64: # res: 4->8->16->32->64
            self.mean = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)
            )
            self.std = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)
            )
            if self.residual: # generate additional mask
                self.mask = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
                    nn.ELU(),
                    nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)
                )
        self.soft_plus = nn.Softplus(beta=1, threshold=20)
        self.apply(weights_init_)

    def forward(self, state, coarse_action=None):
        if coarse_action is None:
            x = self.obs(state)
        else:
            if self.merge:
                state = torch.cat([state, coarse_action], dim=1)
                x = self.obs(state)
            else:
                x1 = self.obs(state)
                x2 = self.obs_a(coarse_action)
                x = torch.cat([x1, x2], dim=1)
        if self.action_res == 4:
            x = x.reshape(x.shape[0], -1)
        mean = self.mean(x).reshape(x.shape[0], -1)
        std = torch.clamp(self.soft_plus(SOFT_PLUS_K * self.std(x) + SOFT_PLUS_B),\
            MIN_STD, MAX_STD).reshape(x.shape[0], -1)
        if self.residual:
            mask = nn.Sigmoid()(self.mask(x).reshape(x.shape[0], -1) + self.bias)
            if self.merge == False:
                mask = 0.25 * mask
            return mean, std, mask
        else:
            return mean, std, None

    def sample(self, state, coarse_action=None):
        for params in self.parameters():
            if torch.isnan(params).any():
                print('Policy has NaN parameters')
        if coarse_action is not None:
            coarse_action = self.upsample(coarse_action)
        mean, std, mask = self.forward(state, coarse_action)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, std, mask
    
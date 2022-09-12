import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scipy.signal
from gym.spaces import Box, Discrete
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from os import listdir
from os.path import isfile, join
import re
import json

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        #self.log_std_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation) # from -inf to inf
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        #log_std = self.log_std_net(obs)
        std = torch.exp(self.log_std)
        #std = torch.exp(log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class HighwayDataset(Dataset):
    def __init__(self, states_list, actions_list):
        self.states_list = states_list
        self.actions_list = actions_list

    def __len__(self):
        return len(self.states_list)

    def __getitem__(self, idx):
        return torch.tensor(self.states_list[idx]), torch.tensor(self.actions_list[idx])

mypath="./scripts/light/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
states_list = []
actions_list = []
for file in onlyfiles:
    with open(mypath+file) as json_file:
        data = json.load(json_file)
        states_list = states_list + data["states_list"][:-1]
        actions_list = actions_list + data["actions_list"]

train_dataset = HighwayDataset(states_list, actions_list)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = MLPGaussianActor(obs_dim=70, act_dim=2, hidden_sizes=(256,256), activation=torch.nn.Tanh)

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())#, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

size = len(train_dataloader.dataset)
epochs = 1
all_losses = []
for t in tqdm(range(epochs)):
    total_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        pi, logp_a = model(X,y)
        loss = loss_fn(logp_a, torch.zeros_like(logp_a))
        #pi, logp_a = model(X)
        #loss = loss_fn(pi.sample(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
    all_losses.append(loss.detach().numpy())
    print(f"loss: {loss:>7f} | mean loss: {np.mean(all_losses)} [{current:>5d}/{size:>5d}]")
    scheduler.step(total_loss)




import gym
import highway_env
import numpy as np

env = gym.make("exit-continuous-v0")
env.configure({
    "manual_control": False,
    "real_time_rendering": True,
    # "random_vehicles_density": True,
    "vehicles_density": 1,  # 1.6,
    "duration": 24,
    "vehicles_count": 15,
    "disable_collision_checks": True,
    # "steering_range": [-np.pi, np.pi],
    # "observation": {
    #            "type": "KinematicFlattenObservation",
    #            "normalize": False,
    #        },

})

env.seed(0)
observation=env.reset()
done = False
i=0
sum_reward = 0
while True:
    while done == False:
        with torch.no_grad():
            pi,_ = model(torch.tensor(observation).float())
            action = pi.sample().numpy()
        observation,reward,done,info = env.step(action)  # with manual control, these actions are ignored
        sum_reward = sum_reward + reward
        env.render()
        i = i+1
        #print(i)
    i=0
    print("sum reward "+str(sum_reward))
    sum_reward = 0
    env.seed(0)
    env.reset()
    done = False
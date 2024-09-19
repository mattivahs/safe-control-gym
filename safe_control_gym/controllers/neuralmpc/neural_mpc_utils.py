"""The Cross Entropy Method implemented with Pytorch.
"""
from typing import Union, Callable, Optional
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from typing import Tuple, Iterator
from collections import deque, namedtuple
import numpy as np

from safe_control_gym.safety_filters.cbfCP.cbf_utils import *

device = 'cpu'# 'cuda:0' if torch.cuda.is_available() else 'cpu'

OBS_SHAPE_ERROR = "obs.shape must be either (obs_dim, ) or (batch_size, obs_dim)!"
AC_BOUND_ERROR = "Both ac_lb and ac_ub must be provided!"


def CostMPC(x, x_ref, u, u_ref, Q, R):
    return 0.5 * (x - x_ref).T @ Q @ (x - x_ref) + 0.5 * (u - u_ref).T @ R @ (u - u_ref)


class LearnedDynamics:
    def __init__(self,
                 env_func,
                 dt=0.02,
                 **config):
        self.buffer = ReplayBuffer(100000)
        self.dataset = RLDataset(self.buffer)

        self.dt = dt
        if env_func.NAME == 'quadrotor':
            self.dyn = Drone2DFull(dt=self.dt).to(device)
        else:
            self.dyn = CartpoleFull(dt=self.dt).to(device)

        self.trainer = Trainer(self.dyn, self.dataset)

    def save(self,
             path
             ):
        '''Saves model params.'''
        self.dyn.save(path)

    def load(self,
             path
             ):
        '''Restores model given checkpoint path.'''
        self.dyn.load(path)

    def train_callback(self, ep, loss):
        self.loss_list.append(loss)
        print("Epoch: {}, Loss: {}".format(ep, loss))

    def learn(self,
              env=None,
              safety_filter=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''
        train_losses = []
        for k in range(self.n_episodes):
            obs, info = env.reset(seed=46)
            ep_ret = 0.
            if k > 0:
                # train
                loss = self.trainer.train(n_epochs=200, batch_size=16, lr=1e-3, cb_fun=self.train_callback,
                                          weight_decay=0.00001)
                train_losses.append(loss)
                print(loss)
            for _ in range(self.n_steps):
                act = torch.tensor(self.select_action(obs, info), device=device, dtype=torch.float32)

                # state s
                s = self.dyn.obs2state(obs)

                if safety_filter is not None:
                    act, _ = safety_filter.certify_action(obs, act.cpu().numpy(), info)
                    act = act.astype(np.float32)
                else:
                    act = act.cpu().numpy()
                # modified for Safe RL, added cost
                obs, reward, terminated, info = env.step(act)

                # state s prime
                sp = self.dyn.obs2state(obs)
                self.buffer.append((s.cpu().numpy(), act, sp.cpu().numpy()))

                ep_ret += reward
                if terminated:
                    break
        loss = self.trainer.train(n_epochs=200, batch_size=16, lr=1e-3, cb_fun=self.train_callback,
                                  weight_decay=0.00001)


class CartpoleFull(nn.Module):
    def __init__(self, hidden_dim=64, hidden_layers=2,
                 m=0.2,  # 0.1
                 M=0.8,  # 1.0
                 g=9.8,
                 dt=0.05,
                 L=0.6,  # 0.5
                 **kwargs):
        super(CartpoleFull, self).__init__()
        self.nx = 5
        self.nu = 1
        self.enforce_rel_deg = True
        self.mlps_f = nn.ModuleList()
        self.mlps_g = nn.ModuleList()
        # state: [x, s(th), c(th), c(th), dx, dth]
        self.nn_input_dim = 5
        self.nn_output_dim_f = 5
        if self.enforce_rel_deg:
            self.nn_output_dim_g = 2  # 2 x 1 matrix
        else:
            self.nn_output_dim_g = 5  # 5 x 1 matrix
        self.mlps_f.append(nn.Linear(self.nn_input_dim, hidden_dim))
        self.mlps_g.append(nn.Linear(self.nn_input_dim, hidden_dim))
        for i in range(hidden_layers - 1):
            self.mlps_f.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlps_g.append(nn.Linear(hidden_dim, hidden_dim))
        self.out_head_f = nn.Linear(hidden_dim, self.nn_output_dim_f)
        self.out_head_g = nn.Linear(hidden_dim, self.nn_output_dim_g)
        self.m = m
        self.M = M
        self.g = g
        self.dt = dt
        self.L = L
        self.f_prior = torch.zeros((2000, 5)).to(device)
        self.g_prior = torch.zeros((2000, 5)).to(device)
        self.g_learned = torch.zeros((2000, 5)).to(device)

    def prior_vectorfields(self, x):
        if x.shape[0] != self.f_prior.shape[0]:
            self.f_prior = torch.zeros_like(x).float().to(device)
            self.g_prior = torch.zeros_like(x).float().to(device)

        ml = self.m * self.L
        mM = self.m + self.M
        self.f_prior[..., 0:1] = x[..., 3:4]
        self.f_prior[..., 1:2] = x[..., 2:3] * x[..., 4:5]
        self.f_prior[..., 2:3] = - x[..., 1:2] * x[..., 4:5]
        denominator = self.L * (4. / 3. - self.m * x[..., 2:3] ** 2 / mM)

        d2theta_f = (self.g * x[..., 1:2] - (ml * x[..., 1:2] * x[..., 2:3] * x[..., 4:5] ** 2) / mM) / denominator
        d2theta_g = - (x[..., 2:3] / mM) / denominator
        self.f_prior[..., 3:4] = (1 / mM) * (ml * x[..., 4:5] ** 2 * x[..., 1:2] - ml * x[..., 2:3] * d2theta_f)
        self.f_prior[..., 4:5] = d2theta_f

        self.g_prior[..., 3:4] = 1 / mM - (ml * x[..., 2:3] / mM) * d2theta_g

        self.g_prior[..., 4:5] = d2theta_g

    def get_f(self, x):
        f_prior = torch.zeros_like(x).to(device)
        ml = self.m * self.L
        mM = self.m + self.M
        f_prior[..., 0:1] = x[..., 3:4]
        f_prior[..., 1:2] = x[..., 2:3] * x[..., 4:5]
        f_prior[..., 2:3] = - x[..., 1:2] * x[..., 4:5]
        denominator = self.L * (4. / 3. - self.m * x[..., 2:3] ** 2 / mM)
        d2theta_f = (self.g * x[..., 1:2] - (ml * x[..., 1:2] * x[..., 2:3] * x[..., 4:5] ** 2) / mM) / denominator
        f_prior[..., 3:4] = (1 / mM) * (ml * x[..., 4:5] ** 2 * x[..., 1:2] - ml * x[..., 2:3] * d2theta_f)
        f_prior[..., 4:5] = d2theta_f

        x_f = x
        for layer in self.mlps_f:
            x_f = F.leaky_relu(layer.to(device)(x_f))

        return f_prior + self.out_head_f.to(device)(x_f)

    def get_g(self, x):
        g_prior = torch.zeros_like(x).float().to(device)
        ml = self.m * self.L
        mM = self.m + self.M
        denominator = self.L * (4. / 3. - self.m * x[..., 2:3] ** 2 / mM)
        d2theta_g = - (x[..., 2:3] / mM) / denominator

        g_prior[..., 3:4] = 1 / mM - (ml * x[..., 2:3] / mM) * d2theta_g

        g_prior[..., 4:5] = d2theta_g

        x_g = x
        for layer in self.mlps_g:
            x_g = F.leaky_relu(layer.to(device)(x_g))
        if self.enforce_rel_deg:
            g_learned = torch.zeros_like(x).to(device)
            g_learned[..., 3:] = self.out_head_g.to(device)(x_g)
        else:
            g_learned = self.out_head_g.to(device)(x_g)

        return (g_prior + g_learned).T

    def forward_nobatch(self, x, act):
        f, g = self.get_f(x), self.get_g(x)

        return x + self.dt * (f + g @ act)

    def forward(self, x, act):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3 and len(act.shape) == 2:
            act = act.unsqueeze(1)

        if len(act.shape) == 1:
            act = act.unsqueeze(-1)

        self.prior_vectorfields(x)

        x_f = x
        x_g = x

        for layer in self.mlps_f:
            x_f = F.leaky_relu(layer(x_f))
        for layer in self.mlps_g:
            x_g = F.leaky_relu(layer(x_g))
        x_f = self.out_head_f(x_f)

        if self.enforce_rel_deg:
            # TODO: don't create zero array every time
            g_learned = torch.zeros_like(x).to(device)
            g_learned[..., 3:] = self.out_head_g.to(device)(x_g)
        else:
            g_learned = self.out_head_g.to(device)(x_g)

        # output of equations of motion, i.e. continuous dynamics model
        x_out = self.f_prior + x_f + (self.g_prior + g_learned) * act
        return x + self.dt * x_out

    def save(self, path):
        torch.save(self.state_dict(), path + "model.pth")

    def load(self, path):
        self.load_state_dict(torch.load(path + "model.pth"))

    def obs2state(self, obs):
        # complete state: x, s(th), c(th), dx, dth
        complete_state = torch.tensor(np.array([
            obs[0],
            np.sin(obs[2]),
            np.cos(obs[2]),
            obs[1],
            obs[3]]
        ), device=device, dtype=torch.float32)
        return complete_state

    def state2obs(self, state):
        # complete state: x, s(th), c(th), dx, dth
        complete_obs = torch.tensor(np.array([
            state[0],
            state[3],
            np.atan2(state[1], state[2]),
            state[4]]
        ), device=device, dtype=torch.float32)
        return complete_obs


class Drone2DFull(nn.Module):
    def __init__(self, hidden_dim=64, hidden_layers=5,
                 m=0.04,
                 g=9.8,
                 Iyy=1.4e-05,
                 dt=0.02,
                 L=0.0397,
                 **kwargs):
        super(Drone2DFull, self).__init__()
        self.nx = 7
        self.nu = 2
        self.enforce_rel_deg = True
        self.mlps_f = nn.ModuleList()
        self.mlps_g = nn.ModuleList()
        # state: [x, z, s(th), c(th), dx, dz, dth]
        self.nn_input_dim = 7
        self.nn_output_dim_f = 7
        if self.enforce_rel_deg:
            self.nn_output_dim_g = 6  # 3 x 2 matrix
        else:
            self.nn_output_dim_g = 14  # 7 x 2 matrix
        self.mlps_f.append(nn.Linear(self.nn_input_dim, hidden_dim))
        self.mlps_g.append(nn.Linear(self.nn_input_dim, hidden_dim))
        for i in range(hidden_layers - 1):
            self.mlps_f.append(nn.Linear(hidden_dim, hidden_dim))
            # self.mlps_f[-1].weight.data.zero_()
            # self.mlps_f[-1].bias.data.zero_()
            self.mlps_g.append(nn.Linear(hidden_dim, hidden_dim))
            # self.mlps_g[-1].weight.data.zero_()
            # self.mlps_g[-1].bias.data.zero_()
        self.out_head_f = nn.Linear(hidden_dim, self.nn_output_dim_f)
        # self.out_head_f.weight.data.zero_()
        # self.out_head_f.bias.data.zero_()
        self.out_head_g = nn.Linear(hidden_dim, self.nn_output_dim_g)
        # self.out_head_g.weight.data.zero_()
        # self.out_head_g.bias.data.zero_()
        self.m = m
        self.g = g
        self.Iyy = Iyy
        self.dt = dt
        self.L = L
        self.f_prior = torch.zeros((1, 7)).float().to(device)
        self.g_prior = torch.zeros((1, 7, 2)).float().to(device)
        self.g_learned = torch.zeros((1, 7, 2)).float().to(device)

    def prior_vectorfields(self, x):
        f_prior = torch.zeros_like(x).float().to(device)
        g_prior = torch.zeros((x.shape[0], 7, 2)).float().to(device)
        f_prior[..., :2] = x[..., 4:6]
        f_prior[..., 2:3] = x[..., 3:4] * x[..., 6:7]
        f_prior[..., 3:4] = - x[..., 2:3] * x[..., 6:7]
        f_prior[..., 5:6] = - self.g

        g_prior[..., 4:, 0] = torch.cat((x[..., 2:3] / self.m,
                                              x[..., 3:4] / self.m,
                                              torch.ones((x.shape[0], 1)).to(device) * (
                                                          - self.L / self.Iyy / np.sqrt(2))), dim=-1)

        g_prior[..., 4:, 1] = torch.cat((x[..., 2:3] / self.m,
                                              x[..., 3:4] / self.m,
                                              torch.ones((x.shape[0], 1)).to(device) * (
                                                          self.L / self.Iyy / np.sqrt(2))), dim=-1)

        return f_prior, g_prior

    def get_f(self, x):
        f_prior = torch.zeros_like(x).to(device)
        f_prior[..., :2] = x[..., 4:6]
        f_prior[..., 2:3] = x[..., 3:4] * x[..., 6:7]
        f_prior[..., 3:4] = - x[..., 2:3] * x[..., 6:7]
        f_prior[..., 5:6] = - self.g

        x_f = x
        for layer in self.mlps_f:
            x_f = F.leaky_relu(layer.to(device)(x_f))

        return f_prior + self.out_head_f.to(device)(x_f)

    def get_g(self, x):
        g_prior = torch.zeros((x.shape[0], 7, 2)).to(device)

        g_prior[..., 4:, 0] = torch.cat((x[..., 2:3] / self.m,
                                         x[..., 3:4] / self.m,
                                         torch.ones((x.shape[0], 1)).to(device) * (- self.L / self.Iyy / np.sqrt(2))),
                                        dim=-1)

        g_prior[..., 4:, 1] = torch.cat((x[..., 2:3] / self.m,
                                         x[..., 3:4] / self.m,
                                         torch.ones((x.shape[0], 1)).to(device) * (self.L / self.Iyy / np.sqrt(2))),
                                        dim=-1)

        x_g = x
        for layer in self.mlps_g:
            x_g = F.leaky_relu(layer.to(device)(x_g))
        if self.enforce_rel_deg:
            g_learned = torch.zeros((x.shape[0], 7, 2)).to(device)
            g_learned[..., 4:, :] = self.out_head_g.to(device)(x_g).reshape((3, 2))
        else:
            g_learned = self.out_head_g.to(device)(x_g).reshape((7, 2))

        return g_prior + g_learned

    def forward_nobatch(self, x, act):
        f, g = self.get_f(x), self.get_g(x)

        return x + self.dt * (f + g @ act)

    def forward(self, state_act):
        x = state_act[..., :self.nx]
        act = state_act[..., self.nx:]
        # if len(x.shape) == 1:
        #     x = x.unsqueeze(0)
        # elif len(x.shape) == 3 and len(act.shape) == 2:
        #     # x = x.squeeze(1)
        #     act = act.unsqueeze(1)
        #
        # if len(act.shape) == 1:
        #     act = act.unsqueeze(0)

        f_prior, g_prior = self.prior_vectorfields(x)

        # x_f = x
        # x_g = x
        # for layer in self.mlps_f:
        #     x_f = F.leaky_relu(layer(x_f))
        # for layer in self.mlps_g:
        #     x_g = F.leaky_relu(layer(x_g))
        # x_f = self.out_head_f(x_f)
        #
        # g_learned = torch.zeros((x.shape[0], self.nx, self.nu)).float().to(device)
        # g_learned[..., 4:, :] = self.out_head_g(x_g).reshape((x.shape[0], 3, 2))

        # output of equations of motion, i.e. continuous dynamics model
        x_out = ((f_prior) +
                  torch.bmm((g_prior), act.unsqueeze(dim=-1)).squeeze(dim=-1))
        return x + self.dt * x_out

    def save(self, path):
        torch.save(self.state_dict(), path + "model.pth")

    def load(self, path):
        self.load_state_dict(torch.load(path + "model.pth"))

    def obs2state(self, obs):
        # complete state: s(th1), c(th1), s(th2), c(th2), dth1, dth2
        complete_state = torch.tensor(np.array([
            obs[0],
            obs[2],
            np.sin(obs[4]),
            np.cos(obs[4]),
            obs[1],
            obs[3],
            obs[5]]
        ), device=device, dtype=torch.float32)
        return complete_state

    def state2obs(self, state):
        # complete state: s(th1), c(th1), s(th2), c(th2), dth1, dth2
        complete_obs = torch.tensor(np.array([
            state[0],
            state[4],
            state[1],
            state[5],
            np.atan2(state[2], state[3]),
            state[6]]
        ), device=device, dtype=torch.float32)
        return complete_obs


class Trainer:
    def __init__(self, model, dataset):
        self.model = model
        self.buffer = dataset

    def train(self, n_epochs=20, batch_size=16, lr=1e-3, cb_fun=None, weight_decay=0.0001):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        overall_loss = 0
        self.model.train()
        # print(len(loader))
        for epoch in range(n_epochs):
            epoch_loss = 0.
            loader = torch.utils.data.DataLoader(self.buffer, batch_size)
            for batch in loader:
                self.optimizer.zero_grad()
                output = self.model(batch[0].to(device), batch[1].to(device))

                loss = torch.nn.MSELoss()(output, batch[2].to(device))
                loss.backward()
                overall_loss += loss.item()
                epoch_loss += loss.item()
                self.optimizer.step()
            if cb_fun is not None:
                cb_fun(epoch, epoch_loss / len(loader))
        return overall_loss / (len(loader) * n_epochs)


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer, sample_size: int = 10) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, new_states = self.buffer.sample(len(self.buffer))
        for i in range(len(states)):
            yield states[i], actions[i], new_states[i]

    def __len__(self) -> int:
        return len(self.buffer)


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "next_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
        )

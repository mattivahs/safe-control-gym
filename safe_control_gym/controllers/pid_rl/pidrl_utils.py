import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from typing import Tuple, Iterator
from collections import deque, namedtuple
import numpy as np


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Drone2DModel(nn.Module):
    def __init__(self, hidden_dim=64, hidden_layers=3, dt=0.02):
        super(Drone2DModel, self).__init__()
        self.mlps_f = nn.ModuleList()
        self.mlps_g = nn.ModuleList()
        # state: [x, z, s(th), c(th), dx, dz, dth]
        self.nn_input_dim = 7
        self.nn_output_dim_f = 3
        self.nn_output_dim_g = 6  # 3 x 2 matrix
        self.mlps_f.append(nn.Linear(self.nn_input_dim, hidden_dim))
        self.mlps_g.append(nn.Linear(self.nn_input_dim, hidden_dim))
        for i in range(hidden_layers - 1):
            self.mlps_f.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlps_g.append(nn.Linear(hidden_dim, hidden_dim))
        self.out_head_f = nn.Linear(hidden_dim, self.nn_output_dim_f)
        self.out_head_g = nn.Linear(hidden_dim, self.nn_output_dim_g)
        self.dt = dt

    def forward(self, x, act):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3 and len(act.shape) == 2:
            # x = x.squeeze(1)
            act = act.unsqueeze(1)

        if len(act.shape) == 1:
            act = act.unsqueeze(0)

        x_old = torch.clone(x).detach()
        x_f = torch.clone(x).detach()
        x_g = torch.clone(x).detach()
        # x = torch.cat((x, act), dim=-1).type(torch.float32)
        for layer in self.mlps_f:
            x_f = F.leaky_relu(layer(x_f))
        for layer in self.mlps_g:
            x_g = F.leaky_relu(layer(x_g))
        x_f = self.out_head_f(x_f)
        x_g = self.out_head_g(x_g)

        # output of equations of motion, i.e. continuous dynamics model
        x = x_f + torch.bmm(x_g.reshape((x_g.shape[0], 3, 2)),act.unsqueeze(dim=-1)).squeeze(dim=-1)

        # trying to use identity: sin(a + b) = sin(a) * cos(b) + cos(a) * sin(b)
        # cos(a + b) = cos(a) * cos(b) - sin(a) * sin(b)
        x_out = x_old.clone()
        dt = self.dt
        # [x, z]
        x_out[..., :2] += dt * x_old[..., 4:6]
        # [s(th), c(th)]
        x_out[..., 2:4] *= torch.cat((torch.cos(dt * x_old[..., 6:7]), torch.cos(dt * x_old[..., 6:7])), dim=-1)
        x_out[..., 2:4] += torch.cat((
            x_old[..., 3:4] * torch.sin(dt * x_old[..., 6:7]),
            - x_old[..., 2:3] * torch.sin(dt * x_old[..., 6:7]),
        ), dim=-1)
        x_out[..., 4:] += dt * x
        return x_out

    def get_vectorfields(self, x):
        x_f = torch.clone(x).detach()
        x_g = torch.clone(x).detach()
        # x = torch.cat((x, act), dim=-1).type(torch.float32)
        for layer in self.mlps_f:
            x_f = F.leaky_relu(layer(x_f))
        for layer in self.mlps_g:
            x_g = F.leaky_relu(layer(x_g))
        x_f = self.out_head_f(x_f)
        x_g = self.out_head_g(x_g)

        return x_f, x_g
    def save(self, path):
        torch.save(self.state_dict(), path + "model.pth")

    def load(self, path):
        self.load_state_dict(torch.load(path + "model.pth"))


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


def obs2state(obs):
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


def state2obs(state):
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

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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

OBS_SHAPE_ERROR = "obs.shape must be either (obs_dim, ) or (batch_size, obs_dim)!"
AC_BOUND_ERROR = "Both ac_lb and ac_ub must be provided!"


def batch_cov(batch: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix with batch dimension.

    Args:
        batch: shape(batch_size, N_samples, data_dim)
    Returns:
        cov: shape(batch_size, data_dim, data_dim)
    """
    num_samples = batch.shape[1]
    centered = batch - batch.mean(dim=1, keepdim=True)
    prods = torch.einsum('bni, bnj->bij', centered, centered)
    cov = prods / (num_samples - 1)
    return cov


class CostFn(nn.Module):
    """An example cost function."""
    def __init__(self, ref=0., model='cartpole'):
        super(CostFn, self).__init__()
        self.reference = ref
        if model == 'quadrotor':
            self.Q = torch.from_numpy(np.diag([100., 100., 10., 10., 0., 0., 0.])).type(torch.float32).to(device)
        elif model == 'cartpole':
            self.Q = torch.from_numpy(np.diag([1, 100., 100., 1., 10.])).type(torch.float32).to(device)

    def update_ref(self, ref):
        self.reference = ref
    def forward(self, obs, ac, timestep=0):
        res = 0 * h_cartpole_batch(obs) + 0*torch.sum(torch.matmul((obs - self.reference[timestep]), self.Q) * (obs - self.reference[timestep]), dim=-1) + 0.1*torch.sum((ac)**2, dim=-1)
        # res = 10 *((torch.sum((obs[..., 0] - self.reference[0])**2, dim=-1) + torch.sum((obs[..., 2] - self.reference[2])**2, dim=-1)
        #        + torch.sum((obs[..., 4:6])**2, dim=-1)) + (torch.sum((obs[..., 1])**2, dim=-1) + torch.sum((obs[..., 3])**2, dim=-1))
        #            + 0*torch.sum((ac)**2, dim=-1))
        return res.flatten()


class TerminalCostFn(nn.Module):
    """An example cost function."""
    def __init__(self, ref=0., model='cartpole'):
        super(TerminalCostFn, self).__init__()
        self.reference = ref
        if model == 'quadrotor':
            self.Q = torch.from_numpy(np.diag([100., 100., 1., 1., 0., 0., 0.])).type(torch.float32).to(device)
        elif model == 'cartpole':
            self.Q = torch.from_numpy(np.diag([5, 10., 10., 1., 0.])).type(torch.float32).to(device)

    def update_ref(self, ref):
        self.reference = ref
    def forward(self, obs):
        res = 1000 * torch.sum(torch.matmul((obs - self.reference), self.Q) * (obs - self.reference), dim=-1)
        return res.flatten()


class CEM:
    """An optimization solver based on Cross Entropy Method under Pytorch.

    This CEM solver supports batch dimension of observations, and can solve receding
    horizon style model predictive control problems. If only one step cost is to be
    considered, just set horizon to be one.
    """

    def __init__(
            self,
            obs_dim: int,
            ac_dim: int,
            dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            ac_lb: torch.Tensor = None,
            ac_ub: torch.Tensor = None,
            terminal_obs_cost: Callable[[torch.Tensor], torch.Tensor] = None,
            num_samples: int = 100,
            num_iterations: int = 3,
            num_elite: int = 10,
            horizon: int = 15,
            init_cov_diag: float = 1.,
            device: str = "cpu",
            **kwargs
    ):
        """Construct all necessary attributes.

        Args:
            obs_dim: the observation (state) dimension.
            ac_dim: the action dimension.
            dynamics_fn: the dynamics function [obs, ac] -> [next_obs]. It is for doing
                rollouts when computing the total cost of a trajectory of H horizon.
            cost_fn: the cost function [obs, ac] -> cost.
            ac_lb: the lower bound of the action space.
            ac_ub: the upper bound of the action space.
            terminal_obs_cost: the function to compute the cost of terminal observation.
            num_samples: the total number of samples generated in each iteration of CEM.
            num_iterations: the number of iterations of the CEM.
            num_elite: the number of top samples to be selected in each iteration of CEM.
            horizon: the horizon of the trajectory to be considered for computing cost.
            init_cov_diag: the diagonal element of the initial covirance of the action.
            device: the device being used by torch.
        """
        self._device = device
        self._dtype = torch.float32

        self.task = kwargs['task']
        self._num_samples = num_samples
        self._horizon = horizon
        self._num_iter = num_iterations
        self._num_elite = num_elite

        self._obs_dim = obs_dim
        self._ac_dim = ac_dim
        self._batch_size = None
        self._ac_lb = ac_lb
        self._ac_ub = ac_ub
        assert self._check_bound(), AC_BOUND_ERROR

        self._dyn = dynamics_fn
        self._cost_fn = cost_fn
        self._terminal_obs_cost = terminal_obs_cost

        self._ac_dist = None
        self._init_cov_diag = init_cov_diag
        self._cov_reg = torch.eye(self._horizon * self._ac_dim, device=self._device,
                                  dtype=self._dtype).unsqueeze(0) * init_cov_diag * 1e-5

    def _init_ac_dist(self):
        assert self._batch_size is not None
        mean = torch.zeros((self._batch_size, self._horizon * self._ac_dim),
                           device=self._device, dtype=self._dtype)
        cov = self._init_cov_diag * torch.eye(self._horizon * self._ac_dim,
                                              device=self._device, dtype=self._dtype).expand(self._batch_size, -1, -1)
        self._ac_dist = MultivariateNormal(mean, covariance_matrix=cov)

    def _check_bound(self):
        if self._ac_lb is not None or self._ac_ub is not None:
            if self._ac_lb is None or self._ac_ub is None:
                return False
            self._ac_lb = self._ac_lb.to(device=self._device)
            self._ac_ub = self._ac_ub.to(device=self._device)
        return True

    def _clamp_ac_samples(self, ac):
        if self._ac_ub is not None:
            ac = torch.clamp(ac, min=self._ac_lb, max=self._ac_ub)
        return ac

    def _slice_current_step(self, t):
        return slice(t * self._ac_dim, (t + 1) * self._ac_dim)

    def _flatten_batch_dim(self, tensor):
        return torch.flatten(tensor, end_dim=1)

    def _recover_batch_dim(self, tensor):
        return tensor.view((self._batch_size, self._num_samples) + tensor.shape[1:])

    def _evaluate_trajectories(self, obs, ac):
        obs = obs.unsqueeze(1).expand(-1, self._num_samples, -1)
        # shape = (batch_size, num_samples, ac_dim)
        obs = self._flatten_batch_dim(obs)
        ac = self._flatten_batch_dim(ac)

        if self._horizon == 1:
            cost_total = torch.squeeze(self._cost_fn(obs, ac))
        else:
            cost_total = torch.zeros(self._batch_size * self._num_samples,
                                     device=self._device, dtype=self._dtype)
            for t in range(self._horizon):
                ac_t = ac[:, self._slice_current_step(t)]
                # obs = obs + 0.02 * torch.from_numpy(np.float32(self._dyn(obs.cpu().numpy(), ac_t.cpu().numpy()))).to(self._device)
                obs = self._dyn(obs, ac_t)
                if self.task == 'stabilization':
                    cost_total += self._cost_fn(obs, ac_t)
                else:
                    cost_total += self._cost_fn(obs, ac_t, t)
            if self._terminal_obs_cost:
                cost_total += self._terminal_obs_cost(obs)
        cost_total = self._recover_batch_dim(cost_total)
        return cost_total

    def _step(self, obs):
        # sample K action trajectories
        ac_samples = self._ac_dist.sample((self._num_samples,))
        # shape(num_samples, batch_size, ac_dim)
        ac_samples = ac_samples.transpose(0, 1)
        # shape(batch_size, num_samples, ac_dim)
        ac_samples = self._clamp_ac_samples(ac_samples)

        cost_total = self._evaluate_trajectories(obs, ac_samples)
        _, topk_idx = torch.topk(cost_total,
                                 self._num_elite, dim=1, largest=False, sorted=False)
        topk_idx = topk_idx.unsqueeze(2).expand(-1, -1, self._horizon * self._ac_dim)
        top_samples = ac_samples.gather(1, topk_idx)

        mean = torch.mean(top_samples, dim=1)
        cov = batch_cov(top_samples)
        cov_rk = torch.linalg.matrix_rank(cov) < self._horizon * self._ac_dim
        cov_rk = cov_rk.view(-1, 1, 1)
        cov += self._cov_reg * cov_rk

        self._ac_dist = MultivariateNormal(mean, covariance_matrix=cov)

        return top_samples

    def solve(
            self, obs: torch.Tensor, get_log_probs: bool = False,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Do the CEM to solve for best actions.

        Args:
            obs: shape(obs_dim, ) or (batch_size, obs_dim) the current observation.
            get_log_probs: whether or not to return the log probabilities of the
                returned actions.
        Returns:
            ac_soln: the action solution. Only the action at the FIRST time step of
                the trajectory is returned.
            log_probs: the log probabilities of the returned actions.
        """
        obs = obs.to(self._device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        assert len(obs.shape) == 2, OBS_SHAPE_ERROR
        # obs.shape == (batch_size, obs_dim)
        self._batch_size = obs.shape[0]
        self._init_ac_dist()

        for _ in range(self._num_iter):
            self._step(obs)
        ac_soln = self._ac_dist.sample((1,))[0]

        if get_log_probs:
            log_probs = self._ac_dist.log_prob(ac_soln).unsqueeze(1)
            return ac_soln[:, :self._ac_dim], log_probs
        return ac_soln[:, :self._ac_dim]


class CartpoleFull(nn.Module):
    def __init__(self, hidden_dim=64, hidden_layers=2,
                 m=0.2, # 0.1
                 M=0.8, # 1.0
                 g=9.8,
                 dt=0.05,
                 L=0.6, # 0.5
                 **kwargs):
        super(CartpoleFull, self).__init__()
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
        denominator = self.L * (4. / 3. - self.m * x[..., 2:3]**2 / mM)

        d2theta_f = (self.g * x[..., 1:2] - (ml * x[..., 1:2] * x[..., 2:3] * x[..., 4:5]**2) / mM) / denominator
        d2theta_g = - (x[..., 2:3] / mM) / denominator
        self.f_prior[..., 3:4] = (1 / mM) * (ml * x[..., 4:5]**2 * x[..., 1:2] - ml * x[..., 2:3] * d2theta_f)
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
        d2theta_f = (self.g * x[..., 1:2] - (ml * x[..., 1:2] * x[..., 2:3] * x[..., 4:5]**2) / mM) / denominator
        f_prior[..., 3:4] = (1 / mM) * (ml * x[..., 4:5]**2 * x[..., 1:2] - ml * x[..., 2:3] * d2theta_f)
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
        self.f_prior = torch.zeros((2000, 7)).to(device)
        self.g_prior = torch.zeros((2000, 7, 2)).to(device)
        self.g_learned = torch.zeros((2000, 7, 2)).to(device)

    def prior_vectorfields(self, x):
        if x.shape[0] != self.f_prior.shape[0]:
            self.f_prior = torch.zeros_like(x).float().to(device)
            self.g_prior = torch.zeros((x.shape[0], 7, 2)).float().to(device)
        self.f_prior[..., :2] = x[..., 4:6]
        self.f_prior[..., 2:3] = x[..., 3:4] * x[..., 6:7]
        self.f_prior[..., 3:4] = - x[..., 2:3] * x[..., 6:7]
        self.f_prior[..., 5:6] = - self.g

        self.g_prior[..., 4:, 0] = torch.cat((x[..., 2:3] / self.m,
                                         x[..., 3:4] / self.m,
                                         torch.ones((x.shape[0], 1)).to(device) * (- self.L / self.Iyy / np.sqrt(2))), dim=-1)

        self.g_prior[..., 4:, 1] = torch.cat((x[..., 2:3] / self.m,
                                         x[..., 3:4] / self.m,
                                         torch.ones((x.shape[0], 1)).to(device) * (self.L / self.Iyy / np.sqrt(2))), dim=-1)

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

    def forward(self, x, act):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3 and len(act.shape) == 2:
            # x = x.squeeze(1)
            act = act.unsqueeze(1)

        if len(act.shape) == 1:
            act = act.unsqueeze(0)

        # f_prior, g_prior = (
        self.prior_vectorfields(x)

        # x_old = torch.clone(x)
        x_f = x
        x_g = x
        # x = torch.cat((x, act), dim=-1).type(torch.float32)
        for layer in self.mlps_f:
            x_f = F.leaky_relu(layer(x_f))
        for layer in self.mlps_g:
            x_g = F.leaky_relu(layer(x_g))
        x_f = self.out_head_f(x_f)
        # x_g = self.out_head_g(x_g)
        if self.enforce_rel_deg:
            g_learned = torch.zeros((x.shape[0], 7, 2)).float().to(device)
            g_learned[..., 4:, :] = self.out_head_g.to(device)(x_g).reshape((x.shape[0], 3, 2))
        else:
            g_learned = self.out_head_g.to(device)(x_g).reshape((x.shape[0], 7, 2))

        # output of equations of motion, i.e. continuous dynamics model
        x_out = ((self.f_prior + x_f) +
                 torch.bmm((self.g_prior + g_learned), act.unsqueeze(dim=-1)).squeeze(dim=-1))
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
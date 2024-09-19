import numpy as np
import torch.nn as nn
import torch
import cvxpy as cp
import bisect
import copy
from math import ceil


def bounding_box_constraints(x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Vectors 'a' corresponding to the normal of the planes
    a = np.zeros((4, 7))
    a[:, 0:2] = np.array([
        [-1, 0],  # for x >= x_min
        [1, 0],  # for x <= x_max
        [0, -1],  # for y >= y_min
        [0, 1] # for y <= y_max
    ])

    # Scalars 'b' corresponding to the shifted values
    b_list = np.array([[
        -x_min,  # for x >= x_min
        x_max,  # for x <= x_max
        -y_min,  # for y >= y_min
        y_max  # for y <= y_max
    ]]).T

    return a, b_list


class SecondOrderCBF():
    def __init__(self, nx, nu, f_dyn, g_dyn, h_candidate, device):
        self.device = device
        self.f = f_dyn
        self.g = g_dyn
        self.h = h_candidate
        self.nx = nx
        self.nu = nu

    def LieDerivatives(self, x):
        if x.grad is not None:
            x.grad.zero_()
        dhdx, = torch.autograd.grad(self.h(x), x)
        d2hdx2 = torch.autograd.functional.hessian(self.h, x).squeeze()
        dfdx = torch.autograd.functional.jacobian(self.f, x).squeeze()
        # dgdx = torch.autograd.functional.jacobian(self.g, x).squeeze()

        # Ldgdxh = torch.einsum('ijk,k->ij', dgdx, dhdx.squeeze())

        Lfh = dhdx @ self.f(x).T
        Lgh = dhdx @ self.g(x)
        Lf2h = (self.f(x) @ (d2hdx2 @ self.f(x).T + (dhdx @ dfdx).T)).squeeze()
        LgLfh = ((d2hdx2 @ self.f(x).T + (dhdx @ dfdx).T).T @ self.g(x)).squeeze()
        return (Lfh.cpu().detach().numpy().squeeze(),
                Lgh.cpu().detach().numpy().squeeze(),
                Lf2h.cpu().detach().numpy(),
                LgLfh.cpu().detach().numpy(),
                dhdx.cpu().detach().numpy())

    def get_control(self, x, u_des, c_pred=0, dt=0.05, ac_lb=None, ac_ub=None):
        Lfh, Lgh, Lf2h, LgLfh, dhdx = self.LieDerivatives(x)
        h = self.h(x).cpu().detach().numpy().squeeze()
        if h < 0:
            print(h)
        u = cp.Variable((self.nu, 1))
        slack = cp.Variable((1, 1))
        # objective = cp.Minimize(cp.norm((u[1] + u[0]) - (u_des[1] + u_des[0])) +
        #                         10.*cp.norm((u[1] - u[0]) - (u_des[1] - u_des[0])) + 1000 * slack**2)
        objective = cp.Minimize(cp.norm(u - u_des) + 10000 * slack ** 2)

        alp1 = 40 # 40
        alp2 = 0.9 * alp1**2 / 4 # 0.9
        # constraints = [Lf2h + LgLfh @ u + alp1 * Lfh + alp2 * h >= 0. + np.linalg.norm(dhdx) * (c_pred / dt) - slack]
        constraints = [Lf2h + LgLfh * u + alp1 * Lfh + alp2 * h >= 0. + np.linalg.norm(dhdx) * (c_pred / dt) - slack]

        if ac_lb is not None:
            constraints.append(u >= -10)#np.expand_dims(ac_lb, axis=0).T)
            constraints.append(u <= 10)#np.expand_dims(ac_ub, axis=0).T)
        prob = cp.Problem(objective, constraints)
        # print(np.linalg.norm(self.dhdx(x)) * c_pred / dt)
        prob.solve(solver=cp.SCS)

        return u.value.reshape(self.nu, 1), slack.value

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dyn = Drone2DFull()
x_range = (-0.5, 0.5)
y_range = (0.8, 1.2)
A, b = bounding_box_constraints(x_range, y_range)
A = torch.from_numpy(A).float().to(device)
b = torch.from_numpy(b).float().to(device)
def h(x):
    # Return a single tensor instead of a list
    return (b[0] - A[0, :] @ x.T).squeeze()

def smooth_min(x):
    gamma = 10
    return - (1 / gamma) * torch.log(torch.sum(torch.exp(- gamma * x)))
def h_offset(x):
    x_offset = x[0][0:2] + torch.hstack((x[0][2] * 0.05, x[0][3] * 0.05))
    return (b[1] - A[1, 0:2] @ x_offset).squeeze()

def h_rectangle(x):
    x_offset = x[0][0:2] + torch.hstack((x[0][2] * 0.01, x[0][3] * 0.01))
    return torch.min(b[2:4].T - A[2:4, 0:2] @ x_offset)

def h_rectangle_batch(x):
    x_offset = x[..., 0:2] + torch.vstack((x[..., 2] * 0.01, x[..., 3] * 0.01)).T
    return torch.min(torch.vstack((torch.min((b[2:4] - A[2:4,0:2] @ x_offset.T).T, 1)[0], torch.zeros(x.shape[0]).to(device))), 0)[0]

def h_circles(x):
    x_offset = x[0][0:2] + torch.hstack((x[0][2] * 0.01, x[0][3] * 0.01))
    circle_1 = torch.norm(x_offset - torch.Tensor([0.7, 1.0]).to(device)) - 0.1
    circle_2 = torch.norm(x_offset - torch.Tensor([-0.7, 1.0]).to(device)) - 0.1
    return smooth_min(torch.hstack((circle_1, circle_2)))

def h_cartpole(x):
    # x_offset = x[..., 0] + (x[..., 1] * 0.5)
    # return 1 - x_offset
    # return 1 - x[..., 0]
    return smooth_min(torch.hstack((0.1 - x[..., 1], x[..., 1] + 0.1)))

def h_cartpole_batch(x):
    return torch.min(1 - x[..., 0], torch.zeros(x.shape[0]).to(device))
    # return smooth_min(torch.hstack((1 - x[..., 0], 0.2 - x[..., 1], x[..., 1] + 0.2)))
class ConformalPredictor:
    def __init__(self, q_init=1, eta=0.1, alpha=0.05):
        self.q = q_init
        self.eta = eta
        self.score = lambda x, y: np.linalg.norm(x - y)
        self.alpha = alpha
        self.predictionSets = [q_init]
        self.scores = []
        self.violations = []
        self.scores_ordered = [q_init]
        self.delta_recursion = alpha

    def GetSet(self, x_meas, x_predicted, timestep=0):
        self.scores.append(self.score(x_predicted, x_meas))
        bisect.insort(self.scores_ordered, self.score(x_predicted, x_meas))
        self.violations.append((self.score(x_predicted, x_meas) > self.q))

        self.delta_recursion += self.eta * (self.alpha - (self.score(x_predicted, x_meas) > self.q))
        self.q = self.scores_ordered[ceil((timestep + 1) * (1-max(0., self.delta_recursion)))]
        self.predictionSets.append(copy.deepcopy(self.q))

        return copy.deepcopy(self.q)

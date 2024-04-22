import numpy as np
import torch.nn as nn
import torch
import cvxpy as cp

from safe_control_gym.controllers.cem.cem_utils import Drone2DFull

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

        Lfh = dhdx @ self.f(x).T
        Lf2h = (self.f(x) @ (d2hdx2 @ self.f(x).T + (dhdx @ dfdx).T)).squeeze()
        LgLfh = ((d2hdx2 @ self.f(x).T + (dhdx @ dfdx).T).T @ self.g(x)).squeeze()
        return Lfh.cpu().detach().numpy().squeeze(), Lf2h.cpu().detach().numpy(), LgLfh.cpu().detach().numpy()

    def get_control(self, x, u_des, c_pred=0, dt=0.02):
        Lfh, Lf2h, LgLfh = self.LieDerivatives(x)
        h = self.h(x).cpu().detach().numpy().squeeze()

        u = cp.Variable((self.nu, 1))
        # slack = cp.Variable((1, 1))
        objective = cp.Minimize(cp.norm(u - u_des.T))

        alp1 = 50
        alp2 = 0.99 * alp1**2 / 4
        constraints = [Lf2h + LgLfh @ u + alp1 * Lfh + alp2 * h >= 0]# + np.linalg.norm(self.dhdx(x)) * c_pred / dt]
        prob = cp.Problem(objective, constraints)
        # print(np.linalg.norm(self.dhdx(x)) * c_pred / dt)
        prob.solve(solver=cp.SCS)

        return u.value.reshape(self.nu, 1), 0.

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dyn = Drone2DFull()
x_range = (-1, 1)
y_range = (-1, 1)
A, b = bounding_box_constraints(x_range, y_range)
A = torch.from_numpy(A).float().to(device)
b = torch.from_numpy(b).float().to(device)
def h(x):
    # Return a single tensor instead of a list
    return (torch.norm(x[0:2]) -0.1) #(b[0] - pow(A[0, :] @ x.T, 3)).squeeze()
model = SecondOrderCBF(7, 2, dyn.get_f, dyn.get_g, h, device)

x = torch.randn(1, 7, device=device, requires_grad=True)
import time
start = time.time()
output = model.LieDerivatives(x)
print("takes " + str(time.time() - start) + "s")
start = time.time()
x = torch.randn(1, 7, device=device, requires_grad=True)
output = model.LieDerivatives(x)
# u, _ = model.get_control(x, np.ones(2).reshape(2, 1))
print("takes " + str(time.time() - start) + "s")
# print(u)
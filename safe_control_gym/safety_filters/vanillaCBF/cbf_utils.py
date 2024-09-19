import numpy as np
import torch.nn as nn
import torch
import cvxpy as cp
import bisect
import copy
from math import ceil
import jax.numpy as jnp
import jax
from cvxopt import matrix, solvers


def bounding_box_constraints(x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Vectors 'a' corresponding to the normal of the planes
    a = np.zeros((4, 6))
    a[:, 0] = np.array([-1, 1, 0, 0])
    a[:, 2] = np.array([0, 0, -1, 1])

    # Scalars 'b' corresponding to the shifted values
    b_list = np.array([[
        -x_min,  # for x >= x_min
        x_max,  # for x <= x_max
        -y_min,  # for y >= y_min
        y_max  # for y <= y_max
    ]]).T

    return a, b_list

class Quadrotor2DModel:
    def __init__(self, gravity=9.8, mass=0.027, inertia=1.4e-5, length=0.0397):
        self.nx = 6
        self.nu = 2
        self.gravity = gravity
        self.m = mass
        self.Iyy = inertia
        self.d = length

    def f(self, x):
        return jnp.array([x[1], 0., x[3], - self.gravity, x[5], 0.])

    def g(self, x):
        return jnp.array([[0., 0.],
                          [jnp.sin(x[4]) / self.m, jnp.sin(x[4]) / self.m],
                          [0., 0.],
                          [jnp.cos(x[4]) / self.m, jnp.cos(x[4]) / self.m],
                          [0., 0.],
                          [- self.d / self.Iyy / jnp.sqrt(2), self.d / self.Iyy / jnp.sqrt(2)]])


class SecondOrderCBF():
    def __init__(self, h_candidate):
        self.use_min = False
        self.dyn = Quadrotor2DModel()
        self.f = self.dyn.f
        self.g = self.dyn.g
        self.h = h_candidate
        self.nx = self.dyn.nx
        self.nu = self.dyn.nu

        if self.use_min:
            self.dhdx = jax.jit(jax.grad(self.h))
            self.d2hdx2 = jax.jit(jax.hessian(self.h))
            self.dfdx = jax.jit(jax.jacfwd(self.f))
            self.LieDerivative = jax.jit(self.LieDerivatives)
        else:
            self.dhdx = []
            self.d2hdx2 = []
            for i, h in enumerate(h_candidate):
                self.dhdx.append(lambda x, h_=h_candidate[i]: jax.jit(jax.grad(h_))(x))
                self.d2hdx2.append(lambda x, h_=h_candidate[i]: jax.jit(jax.hessian(h_))(x))
            self.dfdx = jax.jit(jax.jacfwd(self.f))
            self.LieDerivative = jax.jit(self.LieDerivatives)

    def LieDerivatives(self, x):
        if self.use_min:
            dhdx = self.dhdx(x)
            d2hdx2 = self.d2hdx2(x)
            dfdx = self.dfdx(x)

            Lfh = dhdx @ self.f(x).T
            Lgh = dhdx @ self.g(x)
            Lf2h = (self.f(x) @ (d2hdx2 @ self.f(x).T + (dhdx @ dfdx).T)).squeeze()
            LgLfh = ((d2hdx2 @ self.f(x).T + (dhdx @ dfdx).T).T @ self.g(x)).squeeze()
        else:
            dhdx = [dhdx(x) for dhdx in self.dhdx]
            d2hdx2 = [d2hdx2(x) for d2hdx2 in self.d2hdx2]
            dfdx = self.dfdx(x)

            Lfh = [a @ self.f(x).T for a in dhdx]
            Lgh = [a @ self.g(x) for a in dhdx]
            Lf2h = [(self.f(x) @ (d2hdx2[i] @ self.f(x).T + (dhdx[i] @ dfdx).T)).squeeze() for i in range(len(dhdx))]
            LgLfh = [((d2hdx2[i] @ self.f(x).T + (dhdx[i] @ dfdx).T).T @ self.g(x)).squeeze() for i in range(len(dhdx))]
        return (Lfh, Lgh, Lf2h, LgLfh, dhdx)

    def get_control(self, x, u_des, ac_lb=None, ac_ub=None):
        Lfh, Lgh, Lf2h, LgLfh, dhdx = self.LieDerivative(x)

        alp1 = 20 # 40
        alp2 = 0.99 * alp1**2 / 4 # 0.9
        if self.use_min:
            h = self.h(x)
            G_ineq = matrix(-np.vstack([np.concatenate((LgLfh, np.array([1]))),
                                        np.array([1, 0, 0]),
                                        np.array([-1, 0, 0]),
                                        np.array([0, 1, 0]),
                                        np.array([0, -1, 0])]))
            h_ineq = matrix(-np.array([(- (LgLfh @ np.array([u_des]).T).squeeze(0) - Lf2h - alp1 * Lfh - alp2 * h),
                                      ac_lb[0] - u_des[0],
                                      - ac_ub[0] + u_des[0],
                                      ac_lb[1] - u_des[1],
                                      - ac_ub[1] + u_des[1]]).reshape(-1, 1).astype(np.double))
        else:
            G_ineq = matrix(-np.vstack([np.concatenate((LgLfh[0], np.array([1]))),
                                        np.concatenate((LgLfh[1], np.array([1]))),
                                        np.concatenate((LgLfh[2], np.array([1]))),
                                        np.concatenate((LgLfh[3], np.array([1]))),
                                        np.array([1, 0, 0]),
                                        np.array([-1, 0, 0]),
                                        np.array([0, 1, 0]),
                                        np.array([0, -1, 0])]))
            h_ineq = matrix(-np.array([(- (LgLfh[0] @ np.array([u_des]).T).squeeze(0) - Lf2h[0] - alp1 * Lfh[0] - alp2 * self.h[0](x)),
                                       (- (LgLfh[1] @ np.array([u_des]).T).squeeze(0) - Lf2h[1] - alp1 * Lfh[1] - alp2 *
                                        self.h[1](x)),
                                       (- (LgLfh[2] @ np.array([u_des]).T).squeeze(0) - Lf2h[2] - alp1 * Lfh[2] - alp2 *
                                        self.h[2](x)),
                                       (- (LgLfh[3] @ np.array([u_des]).T).squeeze(0) - Lf2h[3] - alp1 * Lfh[3] - alp2 *
                                        self.h[3](x)),
                                       ac_lb[0] - u_des[0],
                                       - ac_ub[0] + u_des[0],
                                       ac_lb[1] - u_des[1],
                                       - ac_ub[1] + u_des[1]]).reshape(-1, 1).astype(np.double))

        Q = matrix(np.diag(np.concatenate((np.ones(self.nu), np.array([10000])))))
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, matrix(np.zeros(self.nu + 1)), G_ineq, h_ineq)

        return u_des + np.array(sol['x'])[:self.nu, 0], np.array(sol['x'])[-1, 0]


# dyn = Drone2DFull()
x_range = (-0.3, 0.3)
y_range = (0.6, 1.4)
A, b = bounding_box_constraints(x_range, y_range)

def h(x):
    # Return a single tensor instead of a list
    return (b[0] - A[0, :] @ x.T).squeeze()

def smooth_min(x):
    gamma = 10
    return - (1 / gamma) * jnp.log(jnp.sum(jnp.exp(- gamma * x)))
def h_offset(x):
    x_offset = x[0][0:2] + torch.hstack((x[0][2] * 0.05, x[0][3] * 0.05))
    return (b[1] - A[1, 0:2] @ x_offset).squeeze()


def sigma(s):
    k1 = 0.08
    k2 = 1
    k3 = 1
    return k1 * (jnp.exp(-k2 * s + k3) - 1) / (jnp.exp(-k2 * s + k3) + 1)
    # return -k1 * jnp.atan(k2 * s + k3)

def h_rectangle(x):
    # S = jnp.array(
    #     [x[0]+jnp.sin(x[4]) * 0.01, 0, x[2]+jnp.tanh(50*x[1])*jnp.cos(x[4]) * 0.01, 0, 0, 0])
    # return smooth_min(b.T - A @ S)
    S = jnp.array(
        [sigma(jnp.sin(x[4]) * (0.3 - x[0])),
         sigma(jnp.sin(x[4]) * (x[0]-0.3)),
         sigma(jnp.cos(x[4]) * (x[2]-0.6)),
         sigma(jnp.cos(x[4]) * (x[2]-1.4))])
    return smooth_min((b.T - A @ x - S))

def h_rectangle_list():
    S = lambda x: jnp.array(
        [sigma(jnp.sin(x[4]) * (0.3 - x[0])),
         sigma(jnp.sin(x[4]) * (x[0] - 0.3)),
         sigma(jnp.cos(x[4]) * (x[2] - 0.6)),
         sigma(jnp.cos(x[4]) * (x[2] - 1.4))])
    return [lambda y: b[i] - A[i, :] @ y - S(y)[i] for i in range(4)]


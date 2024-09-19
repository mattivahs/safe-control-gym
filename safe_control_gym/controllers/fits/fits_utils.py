'''General MPC utility functions.'''

import jax
import jax.numpy as jnp
from math import floor
import copy


class DifferentiableEuler:
    def __init__(self, dyn, T, dt, control_dt, h_funs=[], J_fun=None, dynamic_J=False):
        # dynamics model
        self.dyn = dyn
        self.ode_jit = jax.jit(self.ode)
        # planning horizon
        self.T = T
        # ode discretization
        self.dt = dt
        # control discretization
        self.control_dt = control_dt

        # ODE solution just in time compilation
        self.odeint = jax.jit(self.integrate_fori)
        # gradient of ode solution wrt initial consitions and inputs
        self.val_grad = jax.value_and_grad(self.odeint)
        self.diff_ode_sol = jax.jit(self.val_grad)

        self.h_jits = []
        self.h_grads = []
        self.dhdss = []
        # constraint functions
        for i, h in enumerate(h_funs):
            self.h_jits.append(jax.jit(h))

            # Capture the current value of i using a default argument
            self.h_grads.append(jax.value_and_grad(lambda x, h_jit=self.h_jits[i]: self.diff_h_fun(x, h_jit)))
            self.dhdss.append(jax.jit(self.h_grads[i]))

        # objective function
        self.J_jit = jax.jit(J_fun)
        if not dynamic_J:
            self.J_grad = jax.grad(self.diff_J_fun)
            self.dJds = jax.jit(self.J_grad)
        else:
            # gradient with respect to state s
            self.J_grad = jax.grad(self.diff_J_fun_dynamic, argnums=0)
            self.dJds = jax.jit(self.J_grad)

        self.x_sol = None

    def diff_h_fun(self, s, h_jit):
        x_sol = self.odeint(s)
        return h_jit(x_sol)

    def diff_J_fun(self, s):
        x_sol = self.odeint(s)
        return self.J_jit(x_sol)

    def diff_J_fun_dynamic(self, s, reference):
        x_sol = self.odeint(s)
        self.xsol = x_sol
        return self.J_jit(x_sol, reference) + 10*jnp.sum(s[self.dyn.nx:] ** 2)


    def ode(self, x, u):
        return self.dyn.f(x) + self.dyn.g(x) @ u

    def integrate(self, s):
        x0 = s[:self.dyn.nx]
        u_seq = s[self.dyn.nx:]
        x_sol = [x0]
        times = [0.]
        for i in range(int(self.T / self.dt)):
            ind = min(u_seq.shape[0] - self.dyn.nu, self.dyn.nu * int(floor(times[-1] / self.control_dt)))
            x_sol.append(x_sol[-1] + self.dt * self.ode_jit(x_sol[-1], u_seq[ind:(ind + self.dyn.nu)]))
            times.append(times[-1] + self.dt)

        return jnp.array(x_sol)

    def integrate_fori(self, s):
        x_sol = jnp.zeros((int(self.T / self.dt), self.dyn.nx))
        x_sol = x_sol.at[0, :].set(s[:self.dyn.nx])
        u_seq = s[self.dyn.nx:]

        def one_step(i, x):
            ind = self.dyn.nu * (jnp.floor((i-1) * self.dt / self.control_dt)).astype(int)
            x = x.at[i].set(x[i-1, :] + self.dt *
                            self.ode_jit(x[i-1, :], jax.lax.dynamic_slice(u_seq, (ind,), (self.dyn.nu,))))

            return x

        out = jax.lax.fori_loop(1, int(self.T / self.dt), one_step, x_sol)

        return out





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


class CartPoleModel:
    def __init__(self, gravity=9.8, mass=0.027, inertia=1.4e-5, length=0.0397):
        self.nx = 4
        self.nu = 1
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

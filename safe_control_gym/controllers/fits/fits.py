'''Optimization as Control Invariant Set (OACIS).'''

from copy import deepcopy

import casadi as cs
import numpy as np
from cvxopt import matrix, solvers
import copy
import time

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.fits.fits_utils import *
from safe_control_gym.controllers.mpc.mpc_utils import reset_constraints, compute_state_rmse

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list


class FITS(BaseController):
    '''MPC with full nonlinear model.'''

    def __init__(
            self,
            env_func,
            horizon: int = 5,
            trajectory_discretization=30,
            alpha_1 = 5,
            alpha_2 = 10,
            warmstart: bool = True,
            soft_constraints: bool = False,
            terminate_run_on_done: bool = True,
            use_min_formulation: bool = False,
            constraint_tol: float = 1e-6,
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            Q_diag: list = [1., 1., 1., 1., 1., 1.],
            **kwargs
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
        '''
        super().__init__(env_func, output_dir=output_dir, use_gpu=use_gpu, seed=seed, **kwargs)

        # Environment
        self.env = env_func()

        # Additional Constraints
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            self.constraints, _, _ = reset_constraints(self.env.constraints.constraints + self.additional_constraints)
        else:
            self.constraints, _, _ = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []

        # Model parameters (need to use our own dynamcis models as we don't use casadi)
        self.model = self.get_prior(self.env)

        self.soft_constraints = soft_constraints
        self.warmstart = warmstart
        self.terminate_run_on_done = terminate_run_on_done
        # min formulation to only use a single constraint
        self.use_min = use_min_formulation

        self.dyn = Quadrotor2DModel()

        # trajectory configuration state
        self.state = jnp.concatenate((jnp.zeros(self.dyn.nx), 0.1*jnp.ones((horizon - 1) * self.dyn.nu)))

        self.dt = self.model.dt
        self.N = horizon
        self.T = (self.N) * self.dt
        self.M = trajectory_discretization


        # actuation constraints
        self.umin = jnp.array(- self.env.constraints.constraints[0].b[:self.dyn.nu])
        self.umax = jnp.array(self.env.constraints.constraints[0].b[self.dyn.nu:])

        self.constraint_functions = []
        for c in self.additional_constraints:
            self.constraint_functions.append(c.sym_func)

        # self.h_x = constraint_fun
        c_funs = [lambda x, c=c: self.h_s(x, c) for c in self.constraint_functions]

        ode_step = self.T / float(self.M)
        self.solver = DifferentiableEuler(self.dyn, self.T, ode_step, self.T / self.N, c_funs, self.J_dynamic, dynamic_J=True)
        # class kappa functions
        self.alp1 = alpha_1
        self.alp2 = alpha_2
        # regularization term
        self.Q = Q_diag

        self.actuation_constraints = jax.jit(self.input_constraints)
        # vector field of s
        self.fs_jit = jax.jit(self.fs)
        self.gs_jit = jax.jit(self.gs)

        # min formulation
        self.softmin = jax.jit(self.smooth_min)
        self.dsoftminds = jax.jit(self.smooth_min_gradient)
        self.min_formulation = jax.jit(self.min_formulation_)

        # Compile functions
        print("### Just-in-time compilation starting ###")
        self.solver.integrate(jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu))
        self.solver.odeint(jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu))
        for dhds in self.solver.dhdss:
            dhds(jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu))
        self.solver.dJds(jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu), jnp.zeros((self.M, self.dyn.nx)))
        self.actuation_constraints(jnp.empty((0, (self.N - 1) * self.dyn.nu)), jnp.empty((0, 1)))
        self.fs_jit(jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu))
        self.gs_jit(jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu))
        if self.use_min and self.additional_constraints:
            self.softmin(jnp.ones(2*((self.N - 1) * self.dyn.nu) + len(self.additional_constraints)))
            self.dsoftminds(jnp.ones(2*((self.N - 1) * self.dyn.nu) + len(self.additional_constraints)),
                            jnp.ones((2*((self.N - 1) * self.dyn.nu) + len(self.additional_constraints),
                                      (self.N - 1) * self.dyn.nu)))
            self.min_formulation(jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu))
        print("### DONE ###")

        self.last_sol = None

    def fs(self, x):
        """
        :param x: concatenated state of initial conditions and input trajectory
        :return: vectorfield f in input trajectory space
        """
        return jnp.concatenate((self.dyn.f(x[:self.dyn.nx]) + self.dyn.g(x[:self.dyn.nx]) @ x[self.dyn.nx:(self.dyn.nx + self.dyn.nu)],
                               jnp.zeros(((self.N - 1) * self.dyn.nu))))

    def gs(self, x):
        return jnp.vstack((np.zeros((self.dyn.nx, (self.N - 1) * self.dyn.nu)),
                          jnp.eye((self.N - 1) * self.dyn.nu)))

    def h_s(self, x_sol, c):
        return jnp.min(-c(x_sol.T))

    def init_solution(self, x0, dt, ref):
        # Optimize initial solution
        for i in range(100):
            self.get_control(x0, dt, ref)

    def J_s(self, x_sol):
        J = 1*(jnp.sum(jnp.linalg.norm(jnp.array([70., 10., 100., 10., 10., 1.]) *  (x_sol[..., :] - jnp.array([0., 0., 1., 0., 0., 0.])), axis=1))
             + 100 * jnp.linalg.norm(jnp.array([70., 10., 100., 10., 10., 1.])*(x_sol[-1, :] - jnp.array([0., 0., 1., 0., 0., 0.]))))
        return 0.1*J#jnp.exp(0.04 * J)

    def J_dynamic(self, x_sol, reference):
        J = (jnp.sum(jnp.linalg.norm(
            jnp.array(self.Q) * (x_sol - jnp.array(reference)), axis=1)))
        return (1 / self.M) * 2 * J  # 0.1 * J

    def input_constraints(self, G_ineq, h_ineq):
        G_ineq = jnp.vstack([G_ineq, jnp.eye((self.N - 1) * self.dyn.nu), - jnp.eye((self.N - 1) * self.dyn.nu)])
        h_ineq = jnp.vstack([h_ineq,
                            (- self.alp2 * (self.state[self.dyn.nx:] - jnp.tile(self.umin, self.N - 1))).reshape(-1, 1),
                            (- self.alp2 * (jnp.tile(self.umax, self.N - 1) - self.state[self.dyn.nx:])).reshape(-1, 1)])
        return G_ineq, h_ineq

    def smooth_min(self, y):
        gamma = 100.0
        return - (1 / gamma) * jnp.log(jnp.sum(jnp.exp(- gamma * y)))

    def smooth_min_gradient(self, y, dhdss):
        gamma = 100.0
        exp_term = jnp.exp(-gamma * y)
        sum_exp = jnp.sum(exp_term)
        gradient = exp_term / sum_exp
        return gradient @ dhdss

    def min_formulation_(self, state):
        h_i, dhds_i = self.solver.dhdss[0](state)

        h_collection = jnp.array(h_i)
        dhds_collection = jnp.array(dhds_i)

        for i in range(1, len(self.additional_constraints)):
            h_i, dhds_i = self.solver.dhdss[i](state)
            h_collection = jnp.hstack([h_collection, h_i])
            dhds_collection = jnp.vstack([dhds_collection, dhds_i])

        dhds_collection = jnp.array(dhds_collection)
        dhds_collection = jnp.vstack([dhds_collection, jnp.hstack([jnp.zeros(((self.N - 1) * self.dyn.nu, self.dyn.nx)), jnp.eye((self.N - 1) * self.dyn.nu)]),
                            - jnp.hstack([jnp.zeros(((self.N - 1) * self.dyn.nu, self.dyn.nx)), jnp.eye((self.N - 1) * self.dyn.nu)])])

        h_collection = jnp.array(h_collection)
        h_collection = jnp.hstack([h_collection,
                         (state[self.dyn.nx:] - jnp.tile(self.umin, self.N - 1)),
                         (jnp.tile(self.umax, self.N - 1) - state[self.dyn.nx:])])

        h = self.softmin(h_collection)
        dhds = self.dsoftminds(h_collection, dhds_collection)

        return h, dhds

    def get_control(self, x, dt_=0.01, reference=None):
        self.state = self.state.at[:self.dyn.nx].set(jnp.array(x))

        G_ineq = jnp.empty((0, (self.N - 1) * self.dyn.nu))
        h_ineq = jnp.empty((0, 1))

        # vector fields of control affine dynamics in s
        f_s = self.fs_jit(self.state)
        g_s = self.gs_jit(self.state)

        if self.use_min:
            if self.additional_constraints:
                h, dhds = self.min_formulation(self.state)

                Lfh = dhds @ f_s
                Lgh = dhds @ g_s
                G_ineq = jnp.vstack([G_ineq, Lgh])
                h_ineq = jnp.vstack([h_ineq, (- 5 * h - Lfh).reshape(-1, 1)])
        else:
            for i in range(len(self.additional_constraints)):
                h, dhds = self.solver.dhdss[i](self.state)

                Lfh = dhds @ f_s
                Lgh = dhds @ g_s
                G_ineq = jnp.vstack([G_ineq, Lgh])
                h_ineq = jnp.vstack([h_ineq, (- self.alp1 * h - Lfh).reshape(-1, 1)])

            G_ineq, h_ineq = self.actuation_constraints(G_ineq, h_ineq)

        G_ineq = matrix(- np.array(G_ineq).astype(np.double))
        h_ineq = matrix(- np.array(h_ineq).astype(np.double).flatten())

        Q = matrix((1 / (((self.N - 1) * self.dyn.nu) / 20)) * 45 * np.diag(np.ones((self.N - 1) * self.dyn.nu)))

        dJds = self.solver.dJds(self.state, reference)
        p = matrix(np.array(dJds @ g_s).astype(np.double))

        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G_ineq, h_ineq)

        # update control trajectory
        v = np.array(sol['x'])[:, 0]
        u_out = copy.copy(self.state[self.dyn.nx:self.dyn.nx + self.dyn.nu])
        x_out = self.solver.odeint(self.state)
        self.state = self.state.at[self.dyn.nx:].set(self.state[self.dyn.nx:] + v * dt_)

        return u_out, x_out


    def add_constraints(self,
                        constraints
                        ):
        '''Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        '''
        self.constraints, _, _ = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self,
                           constraints
                           ):
        '''Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        '''
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, _, _ = reset_constraints(old_constraints_list)

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def reset(self):
        '''Prepares for training or evaluation.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

        self.setup_results_dict()

    def compute_initial_guess(self, init_state, goal_states, x_lin, u_lin):
        pass
    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        '''

        # Assign reference trajectory within horizon.
        goal_states = self.get_references()

        if self.traj_step == 0 and self.warmstart:
            self.init_solution(obs, self.dt, goal_states.T)

        if self.mode == 'tracking':
            self.traj_step += 1

        # Solve the OACIS problem.
        start = time.time()
        action, planned_traj = self.get_control(obs, self.dt, goal_states.T)
        t_comp = time.time() - start
        print(t_comp)
        self.results_dict['t_wall'].append(t_comp)
        self.results_dict['horizon_states'].append(deepcopy(planned_traj))
        # self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))

        return action

    def get_references(self):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.M))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.traj.shape[-1])
            # end = min(self.traj_step + self.N, self.traj.shape[-1])

            traj_dt = self.T / self.M
            last_control_ind = 0
            goal_states = self.traj[:, start]
            for i in range(1, self.M):
                if i * traj_dt > last_control_ind * self.dt:
                    last_control_ind = min(last_control_ind + 1, self.traj.shape[-1] - 1 - start)
                    goal_states = np.vstack([goal_states, self.traj[:, start + last_control_ind]])
                else:
                    goal_states = np.vstack([goal_states, self.traj[:, start + last_control_ind]])
            # remain = max(0, self.M - (end - start))
            # goal_states = np.concatenate([
            #     self.traj[:, start:end],
            #     np.tile(self.traj[:, -1:], (1, remain))
            # ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states.T  # (nx, M).

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {'obs': [],
                             'reward': [],
                             'done': [],
                             'info': [],
                             'action': [],
                             'horizon_inputs': [],
                             'horizon_states': [],
                             'goal_states': [],
                             'frames': [],
                             'state_mse': [],
                             'common_cost': [],
                             'state': [],
                             'state_error': [],
                             't_wall': []
                             }

    def run(self,
            env=None,
            render=False,
            logging=False,
            max_steps=None,
            terminate_run_on_done=None
            ):
        '''Runs evaluation with current policy.

        Args:
            render (bool): if to do real-time rendering.
            logging (bool): if to log on terminal.

        Returns:
            dict: evaluation statisitcs, rendered frames.
        '''
        if env is None:
            env = self.env
        if terminate_run_on_done is None:
            terminate_run_on_done = self.terminate_run_on_done


        obs = env.reset()
        print('Init State:')
        print(obs)

        ep_returns, ep_lengths = [], []
        frames = []
        self.setup_results_dict()
        self.results_dict['obs'].append(obs)
        self.results_dict['state'].append(env.state)
        i = 0
        if env.TASK == Task.STABILIZATION:
            if max_steps is None:
                MAX_STEPS = int(env.CTRL_FREQ * env.EPISODE_LEN_SEC)
            else:
                MAX_STEPS = max_steps
        elif env.TASK == Task.TRAJ_TRACKING:
            if max_steps is None:
                MAX_STEPS = self.traj.shape[1]
            else:
                MAX_STEPS = max_steps
        else:
            raise Exception('Undefined Task')
        self.terminate_loop = False
        done = False
        common_metric = 0
        while not (done and terminate_run_on_done) and i < MAX_STEPS and not (self.terminate_loop):
            action = self.select_action(obs)
            if self.terminate_loop:
                print('Infeasible MPC Problem')
                break
            # Repeat input for more efficient control.
            obs, reward, done, info = env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)
            self.results_dict['state'].append(env.state)
            self.results_dict['state_mse'].append(info['mse'])
            # self.results_dict['state_error'].append(env.state - env.X_GOAL[i,:])

            common_metric += info['mse']
            print(i, '-th step.')
            print(action)
            print(obs)
            print(reward)
            print(done)
            print(info)
            print()
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            i += 1
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        if logging:
            msg = '****** Evaluation ******\n'
            msg += 'eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n'.format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
        if len(frames) != 0:
            self.results_dict['frames'] = frames
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['state'] = np.vstack(self.results_dict['state'])
        try:
            self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
            self.results_dict['action'] = np.vstack(self.results_dict['action'])
            self.results_dict['full_traj_common_cost'] = common_metric
            self.results_dict['total_rmse_state_error'] = compute_state_rmse(self.results_dict['state'])
            self.results_dict['total_rmse_obs_error'] = compute_state_rmse(self.results_dict['obs'])
        except ValueError:
            raise Exception('[ERROR] mpc.run().py: MPC could not find a solution for the first step given the initial conditions. '
                            'Check to make sure initial conditions are feasible.')
        return deepcopy(self.results_dict)

    def reset_before_run(self, obs, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.reset()

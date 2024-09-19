'''Dynamics learning with PID control class for Crazyflies.

Represent Dynamics as neural net and learn online from data. Data is simply generated by existing PID controller.
'''

import time

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.benchmark_env import Environment, Task
from safe_control_gym.controllers.cem.cem_utils import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import casadi as cs


class CEMRL(BaseController):
    '''PID RL class'''

    def __init__(self,
                 env_func=None,
                 n_episodes=3,
                 n_steps=300,
                 **config
                 ):
        '''Common control classes __init__ method.'''

        super().__init__(env_func, **config)

        self.buffer = ReplayBuffer(100000)
        self.dataset = RLDataset(self.buffer)
        self.env = env_func
        # self.env.reset()
        self.model = self.get_prior(self.env)
        self.task = config['task']
        self.dt = 1 / config['ctrl_freq']

        # self.prior_dyn_jit = cs.Function('f', [obs,act], [self.prior_dyn_b(obs, act)], dict(jit=True,
        #                                                                                     compiler="shell",
        #                                                                                     jit_options=dict(compiler="gcc",
        #                                                                                                      flags="-Ofast")))

        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.loss_list = []
        self.horizon = 15

        self.reference = np.empty(0)

        if config['model'] == 'quadrotor':
            self.dyn = Drone2DFull(dt=self.dt).to(device)
        else:
            self.dyn = CartpoleFull(dt=self.dt).to(device)

        self.trainer = Trainer(self.dyn, self.dataset)
        # CEM agent
        self.cost_fn = CostFn()
        self.terminal_cost = TerminalCostFn()
        self.agent = CEM(self.env.observation_space.shape[0] + 1,
                         ac_dim=self.env.action_dim,
                         dynamics_fn=self.dyn,
                         cost_fn=self.cost_fn,
                         terminal_obs_cost=self.terminal_cost,
                         ac_lb=torch.Tensor(self.env.action_space.low)[0],
                         ac_ub=torch.Tensor(self.env.action_space.high)[0],
                         num_samples=10000,
                         num_iterations=5,
                         num_elite=1500,
                         horizon=self.horizon,
                         init_cov_diag=5, #0.2
                         device=device,
                         **config)


    def prior_dyn_b(self, obs, act):
        out = []
        for i in range(1000):
            out.append(self.prior_dyn(obs[i, :], act[i, :]))

        out = cs.hcat(out).T
        return out

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
        print(loss)

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
        with torch.no_grad():
            s = self.dyn.obs2state(obs)
            # s = torch.from_numpy(obs)
            if self.task == 'stabilization':
                if 'x_reference' in info.keys():
                    self.cost_fn.update_ref(self.dyn.obs2state(info['x_reference']).unsqueeze(0))
                    self.terminal_cost.update_ref(self.dyn.obs2state(info['x_reference']).unsqueeze(0))
            elif self.task == 'traj_tracking':
                if 'x_reference' in info.keys():
                    self.reference = self.dyn.obs2state(info['x_reference'].T).T
                ref = self.reference[info['current_step']:min(info['current_step']+self.horizon, self.reference.shape[0]-1), :]
                while ref.shape[0] < self.horizon:
                    ref = torch.cat((ref, ref[-1,:].unsqueeze(0)))
                self.cost_fn.update_ref(ref)
                self.terminal_cost.update_ref(self.reference[min(info['current_step']+self.horizon, self.reference.shape[0]-1), :])

            start_time = time.time()
            a = self.agent.solve(s, get_log_probs=False)
            # print(a.cpu().numpy().squeeze())
            # print("CEM takes {} seconds".format(time.time() - start_time))
        return a.cpu().numpy().squeeze()

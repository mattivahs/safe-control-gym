'''Dynamics learning with PID control class for Crazyflies.

Represent Dynamics as neural net and learn online from data. Data is simply generated by existing PID controller.
'''

import time

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.benchmark_env import Environment, Task
from safe_control_gym.controllers.cem.cem_utils import *
import l4casadi as l4c

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import casadi as cs


class CEMRL(BaseController):
    '''PID RL class'''

    def __init__(self,
                 env_func=None,
                 n_episodes=3,
                 n_steps=150,
                 **config
                 ):
        '''Common control classes __init__ method.'''

        super().__init__(env_func, **config)

        self.residual_dyn = Drone2DModel().to(device)
        self.buffer = ReplayBuffer(100000)
        self.dataset = RLDataset(self.buffer)
        self.env = env_func()
        self.env.reset()
        self.model = self.get_prior(self.env)
        self.task = config['task']

        self.prior_dyn = Drone2DPrior().to(device)
        # self.prior_dyn_jit = cs.Function('f', [obs,act], [self.prior_dyn_b(obs, act)], dict(jit=True,
        #                                                                                     compiler="shell",
        #                                                                                     jit_options=dict(compiler="gcc",
        #                                                                                                      flags="-Ofast")))

        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.loss_list = []
        self.horizon = 15

        self.reference = np.empty(0)

        self.dyn = Drone2DFull().to(device)
        self.trainer = Trainer(self.dyn, self.dataset)
        # CEM agent
        self.cost_fn = CostFn()
        self.terminal_cost = TerminalCostFn()
        self.agent = CEM(obs_dim=7,#self.dynModel.nn_input_dim,
                         ac_dim=self.env.action_space.shape[0],
                         dynamics_fn=self.dyn,
                         cost_fn=self.cost_fn,
                         terminal_obs_cost=self.terminal_cost,
                         ac_lb=torch.Tensor(self.env.action_space.low)[0],
                         ac_ub=torch.Tensor(self.env.action_space.high)[0],
                         num_samples=2000,
                         num_iterations=5,
                         num_elite=100,
                         horizon=self.horizon,
                         init_cov_diag=0.2,
                         device=device,
                         **config)


    def prior_dyn_b(self, obs, act):
        out = []
        for i in range(1000):
            out.append(self.prior_dyn(obs[i, :], act[i, :]))

        out = cs.hcat(out).T
        return out

    def full_model(self, obs, act):
        return self.prior_dyn(obs, act)

    def save(self,
             path
             ):
        '''Saves model params.'''
        self.residual_dyn.save(path)

    def load(self,
             path
             ):
        '''Restores model given checkpoint path.'''
        self.residual_dyn.load(path)

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
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''
        train_losses = []
        for k in range(self.n_episodes):
            obs, info = env.reset()
            ep_ret = 0.
            if k > 0 and False:
                # train
                loss = self.trainer.train()
                train_losses.append(loss)
                print(loss)
            for _ in range(self.n_steps):
                act = torch.tensor(self.select_action(obs, info), device=device, dtype=torch.float32)

                # state s
                s = obs2state(obs)

                # modified for Safe RL, added cost
                obs, reward, terminated, info = env.step(act.cpu())

                # state s prime
                sp = obs2state(obs)
                self.buffer.append((s.cpu().numpy(), act.cpu(), sp.cpu().numpy()))

                ep_ret += reward
                if terminated:
                    observation, info = env.reset()
        loss = self.trainer.train(n_epochs=500, batch_size=16, lr=1e-3, cb_fun=self.train_callback,
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
        s = obs2state(obs)
        # s = torch.from_numpy(obs)
        if self.task == 'stabilization':
            if 'x_reference' in info.keys():
                self.cost_fn.update_ref(obs2state(info['x_reference']))
                self.terminal_cost.update_ref(obs2state(info['x_reference']))
        elif self.task == 'traj_tracking':
            if 'x_reference' in info.keys():
                self.reference = obs2state(info['x_reference'].T).T
            ref = self.reference[info['current_step']:min(info['current_step']+self.horizon, self.reference.shape[0]-1), :]
            while ref.shape[0] < self.horizon:
                ref = torch.cat((ref, ref[-1,:].unsqueeze(0)))
            self.cost_fn.update_ref(ref)
            self.terminal_cost.update_ref(self.reference[min(info['current_step']+self.horizon, self.reference.shape[0]-1), :])

        start_time = time.time()
        a = self.agent.solve(s, get_log_probs=False)
        # print("calc takes {} seconds".format(time.time() - start_time))
        return a.cpu().numpy().squeeze()

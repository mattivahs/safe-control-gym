'''Control barrier function (CBF) quadratic programming (QP) safety filter.

Reference:
    * [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199)
'''

from typing import Tuple

import numpy as np
import time

from safe_control_gym.safety_filters.base_safety_filter import BaseSafetyFilter
from safe_control_gym.safety_filters.cbfCP.cbf_utils import *
from safe_control_gym.controllers.cem.cem_utils import Drone2DFull

import torch
import torch.nn as nn

class CBF(BaseSafetyFilter):
    '''Control Barrier Function Class.'''

    def __init__(self,
                 env_func,
                 model: nn.Module = None,
                 **config):
        '''
        CBF-QP Safety Filter: The CBF's superlevel set defines a positively control invariant set.
        A QP based on the CBF's Lie derivative with respect to the dynamics allows to filter arbitrary control
        inputs to keep the system inside the CBF's superlevel set.

        Args:
            env_func (partial BenchmarkEnv): Functionalized initialization of the environment.
            slope (float): The slope of the linear function in the CBF.
            soft_constrainted (bool): Whether to use soft or hard constraints.
            slack_weight (float): The weight of the slack in the optimization.
            slack_tolerance (float): How high the slack can be in the optimization.
        '''

        super().__init__(env_func=env_func, **config)
        self.env = self.env_func

        self.setup_results_dict()


        self.dynamics = model

        self.filter = SecondOrderCBF(self.env.observation_space.shape[0]+1, self.env.action_dim, self.dynamics.get_f, self.dynamics.get_g, h_cartpole, device)

        self.predicted_state = None
        self.dt = 1 / config['ctrl_freq']

        q_init = 1.
        eta = 0.05
        alpha = 0.2
        self.ACP = ConformalPredictor(q_init, eta, alpha)

        self.reset()

    def certify_action(self,
                       current_state: np.ndarray,
                       uncertified_action: np.ndarray,
                       info: dict = None
                       ) -> Tuple[np.ndarray, bool]:
        '''Determines a safe action from the current state and proposed action.

        Args:
            current_state (np.ndarray): Current state/observation.
            uncertified_action (np.ndarray): The uncertified_controller's action.
            info (dict): The info at this timestep.

        Returns:
            certified_action (np.ndarray): The certified action.
            success (bool): Whether the safety filtering was successful or not.
        '''
        # Run ACP
        start = time.time()
        uncertified_action = np.clip(uncertified_action, self.env.physical_action_bounds[0], self.env.physical_action_bounds[1])
        self.results_dict['uncertified_action'].append(uncertified_action)
        # certified_action, success = self.solve_optimization(current_state, uncertified_action)
        x = self.dynamics.obs2state(current_state)
        if self.predicted_state is None:
            self.predicted_state = x
        q = self.ACP.GetSet(x.cpu().detach().numpy(), self.predicted_state.cpu().detach().numpy(), info['current_step'])
        # print("predicted: " + str(self.predicted_state.cpu().detach().numpy().tolist()))
        # print("Measured: " + str(x.cpu().detach().numpy().tolist()))
        # print("Score: " + str(self.ACP.scores[-1]))
        print("timestep: " + str(info['current_step']) + ", CP: " + str(q))
        x.requires_grad = True
        # certified_action, success = self.filter.get_control(x.unsqueeze(0), uncertified_action.reshape((2, 1)),
        #                                                     c_pred=q, ac_lb=self.env.physical_action_bounds[0],
        #                                                     ac_ub=self.env.physical_action_bounds[1])
        certified_action, success = self.filter.get_control(x.unsqueeze(0), uncertified_action, dt=self.dt,
                                                            c_pred=q, ac_lb=self.env.physical_action_bounds[0],
                                                            ac_ub=self.env.physical_action_bounds[1])
        print("slack: " + str(success))
        certified_action = np.clip(certified_action.T, self.env.physical_action_bounds[0], self.env.physical_action_bounds[1])

        self.results_dict['feasible'].append(success)
        certified_action = np.squeeze(np.array(certified_action))
        self.results_dict['certified_action'].append(certified_action)
        self.results_dict['correction'].append(np.linalg.norm(certified_action - uncertified_action))
        self.results_dict['h_val'].append(self.filter.h(x.unsqueeze(0)).cpu().detach().numpy())
        self.results_dict['prediction_regions'].append(q)
        with torch.no_grad():
            act = torch.from_numpy(certified_action).float().to(device)
            if len(act.shape) ==  0:
                act = act.unsqueeze(0)
            self.predicted_state = self.dynamics.forward_nobatch(x.unsqueeze(0), act).squeeze()
        # print("Safety filter takes :" + str(time.time() - start) + " s.")
        return certified_action, success

    def is_cbf(self,
               num_points: int = 100,
               tolerance: float = 0.01
               ) -> Tuple[bool, list]:
        '''
        Check if the provided CBF candidate is actually a CBF for the system using a gridded approach.

        Args:
            num_points (int): The number of points in each dimension to check.
            tolerance (float): The tolerance of the condition outside the superlevel set.

        Returns:
            valid_cbf (bool): Whether the provided CBF candidate is valid.
            infeasible_states (list): List of all states for which the QP is infeasible.
        '''
        pass
        # valid_cbf = False
        # epsilon = 1e-6
        #
        # # Select the states to check the CBF condition
        # max_bounds = np.array(self.state_limits)
        # # Add some tolerance to the bounds to also check the condition outside of the superlevel set
        # max_bounds += tolerance
        # min_bounds = -max_bounds
        #
        # # state dimension and input dimension
        # nx, nu = self.model.nx, self.model.nu
        #
        # # Make sure that every vertex is checked
        # num_points = max(2 * nx, num_points + num_points % (2 * nx))
        # num_points_per_dim = num_points // nx
        #
        # # Create the lists of states to check
        # states_to_sample = [np.linspace(min_bounds[i], max_bounds[i], num_points_per_dim) for i in range(nx)]
        # states_to_check = cartesian_product(*states_to_sample)
        #
        # # Set dummy control input
        # control_input = np.ones((nu, 1))
        #
        # num_infeasible = 0
        # num_infeasible_states_inside_set = 0
        # infeasible_states = []
        #
        # # Check if the optimization problem is feasible for every considered state
        # for state in states_to_check:
        #     # Certify action
        #     _, success = self.certify_action(state, control_input)
        #
        #     if not success:
        #         infeasible_states.append(state)
        #         num_infeasible += 1
        #         barrier_at_x = self.cbf(X=state)['cbf']
        #
        #         # Check if the infeasible point is inside or outside the superlevel set. Note that the sampled region makes up a
        #         # box, but the superlevel set is not. The superlevel set only needs to be contained inside the box.
        #         if barrier_at_x > 0.0 + epsilon:
        #             num_infeasible_states_inside_set += 1

        # print('Number of infeasible states:', num_infeasible)
        # print('Number of infeasible states inside superlevel set:', num_infeasible_states_inside_set)
        #
        # if num_infeasible_states_inside_set > 0:
        #     valid_cbf = False
        #     print('The provided CBF candidate is not a valid CBF.')
        # elif num_infeasible > 0:
        #     valid_cbf = True
        #     print('The provided CBF candidate is a valid CBF inside its superlevel set for the checked states. '
        #           'Consider increasing the sampling resolution to get a more precise evaluation. '
        #           'The CBF is not valid on the entire provided domain. Consider softening the CBF constraint by '
        #           'setting \'soft_constraint: True\' inside the config.')
        # else:
        #     valid_cbf = True
        #     print('The provided CBF candidate is a valid CBF for the checked states. '
        #           'Consider increasing the sampling resolution to get a more precise evaluation.')
        #
        # return valid_cbf, infeasible_states

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {}
        self.results_dict['feasible'] = []
        self.results_dict['uncertified_action'] = []
        self.results_dict['certified_action'] = []
        self.results_dict['correction'] = []
        self.results_dict['h_val'] = []
        self.results_dict['prediction_regions'] = []

    def reset(self):
        '''Resets the safety filter.'''
        self.model = self.get_prior(self.env, self.prior_info)
        # self.env.reset()
        self.setup_results_dict()

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

'''Control barrier function (CBF) quadratic programming (QP) safety filter.

Reference:
    * [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199)
'''

from typing import Tuple

import numpy as np
import time

from safe_control_gym.safety_filters.base_safety_filter import BaseSafetyFilter
from safe_control_gym.safety_filters.vanillaCBF.cbf_utils import *

import torch
import torch.nn as nn

class CBF(BaseSafetyFilter):
    '''Control Barrier Function Class.'''

    def __init__(self,
                 env_func,
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
        self.env = env_func()

        self.setup_results_dict()

        S = lambda x: jnp.array(
            [sigma(jnp.sin(x[4]) * (0.3 - x[0])),
             sigma(jnp.sin(x[4]) * (x[0] - 0.3)),
             sigma(jnp.cos(x[4]) * (x[2] - 0.6)),
             sigma(jnp.cos(x[4]) * (x[2] - 1.4))])
        cbf_list =  [lambda y, i=i: (b[i] - A[i, :] @ y - S(y)[i])[0] for i in range(4)]

        self.filter = SecondOrderCBF(cbf_list)

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
        uncertified_action = np.clip(uncertified_action, self.env.physical_action_bounds[0], self.env.physical_action_bounds[1])
        self.results_dict['uncertified_action'].append(uncertified_action)
        # certified_action, success = self.solve_optimization(current_state, uncertified_action)

        start = time.time()
        certified_action, success = self.filter.get_control(current_state, uncertified_action,ac_lb=self.env.physical_action_bounds[0],
                                                            ac_ub=self.env.physical_action_bounds[1])
        t_comp = time.time() - start
        print(t_comp)
        certified_action = np.clip(certified_action.T, self.env.physical_action_bounds[0], self.env.physical_action_bounds[1])

        self.results_dict['t_wall'].append(t_comp)
        self.results_dict['feasible'].append(success)
        certified_action = np.squeeze(np.array(certified_action))
        self.results_dict['certified_action'].append(certified_action)
        self.results_dict['correction'].append(np.linalg.norm(certified_action - uncertified_action))
        if self.filter.use_min:
            self.results_dict['h_val'].append(self.filter.h(current_state))
        else:
            self.results_dict['h_val'].append([self.filter.h[i](current_state) for i in range(len(self.filter.h))])
        return certified_action, success

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {}
        self.results_dict['feasible'] = []
        self.results_dict['uncertified_action'] = []
        self.results_dict['certified_action'] = []
        self.results_dict['correction'] = []
        self.results_dict['h_val'] = []
        self.results_dict['prediction_regions'] = []
        self.results_dict['t_wall'] = []

    def reset(self):
        '''Resets the safety filter.'''
        # self.env.reset()
        self.setup_results_dict()

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

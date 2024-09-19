'''An MPC and Linear MPC example.'''

import os
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from matplotlib.animation import FuncAnimation, PillowWriter


def run(gui=True, n_episodes=1, n_steps=None, save_data=True):
    '''The main function running MPC and Linear MPC experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    random_env = env_func(gui=False)

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment.
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state, _ = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
        experiment.launch_training()

        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        if gui:
            post_analysis(trajs_data['obs'][0], trajs_data['action'][0], ctrl.env)
            generate_trajectory_gif(trajs_data['obs'][0], trajs_data['controller_data'][0]['horizon_states'][0],
                                    trajs_data['controller_data'][0]['goal_states'][0], ctrl.env)

        # Close environments
        static_env.close()
        static_train_env.close()

        # Merge in new trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)

    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics, 'reference': ctrl.env.X_GOAL}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))


def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine iLQR's success.

    Args:
        state_stack (ndarray): The list of observations of iLQR in the latest run.
        input_stack (ndarray): The list of inputs of iLQR in the latest run.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx)
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    # Plot xz trajectory
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.plot(np.array(state_stack).T[0, :], np.array(state_stack).T[2, :], label='Trajectory')
    rect = patches.Rectangle((-0.3, 0.6), 0.6, 0.8, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()

def generate_trajectory_gif(state_stack, plan_stack, goals, env, filename='trajectory.gif'):
    # Initialize the figure and axis
    fig, ax = plt.subplots()

    # Set the limits of the plot
    ax.set_xlim(np.min(np.array(state_stack).T[0, :]) - 0.2, np.max(np.array(state_stack).T[0, :]) + 0.2)
    ax.set_ylim(np.min(np.array(state_stack).T[2, :]) - 0.2, np.max(np.array(state_stack).T[2, :]) + 0.2)

    # Plot the goal positions
    ax.plot(env.X_GOAL[:, 0], env.X_GOAL[:, 2], linestyle='dotted', color='black')

    # Add the rectangle
    rect = patches.Rectangle((-0.35, 0.55), 0.7, 0.9, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Initialize the circle object which will be updated
    goal_line, = ax.plot([], [], label='Trajectory', color='green', linewidth=3)

    # Initialize the line object which will be updated
    line, = ax.plot([], [], label='Trajectory', color='blue')
    plan_line, = ax.plot([], [], label='Trajectory', color='c')

    # Initialize the drone representation as a line and two circles
    drone_line, = ax.plot([], [], color='black')
    drone_circle1 = patches.Circle((0, 0), 0.01, color='black')
    drone_circle2 = patches.Circle((0, 0), 0.01, color='black')
    ax.add_patch(drone_circle1)
    ax.add_patch(drone_circle2)

    # Update function for the animation
    def update(frame):
        line.set_data(np.array(state_stack).T[0, :frame], np.array(state_stack).T[2, :frame])
        plan_line.set_data(np.array(plan_stack).T[:, 0, frame], np.array(plan_stack).T[:, 2, frame])
        goal_line.set_data(goals[frame, 0, :], goals[frame, 2, :])

        # Update the drone position and orientation
        x = np.array(state_stack).T[0, frame]
        z = np.array(state_stack).T[2, frame]
        theta = np.array(state_stack).T[4, frame]

        # Drone endpoints
        dx = 0.05 * np.cos(-theta)
        dz = 0.05 * np.sin(-theta)

        drone_line.set_data([x - dx, x + dx], [z - dz, z + dz])
        drone_circle1.center = (x - dx, z - dz)
        drone_circle2.center = (x + dx, z + dz)
        return line, plan_line, goal_line, drone_line, drone_circle1, drone_circle2

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(state_stack) - 1, blit=True)

    # Save the animation as a GIF
    ani.save(filename, writer=PillowWriter(fps=50))

    plt.show()


def wrap2pi_vec(angle_vec):
    '''Wraps a vector of angles between -pi and pi.

    Args:
        angle_vec (ndarray): A vector of angles.
    '''
    for k, angle in enumerate(angle_vec):
        while angle > np.pi:
            angle -= np.pi
        while angle <= -np.pi:
            angle += np.pi
        angle_vec[k] = angle
    return angle_vec


if __name__ == '__main__':
    run()

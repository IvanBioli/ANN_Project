import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
import os


def epsilon_greedy_action(grid, Q, epsilon):
    """
    Performs a (1-epsilon)-greedy action starting from a given state and given Q-values
    :param grid: current state
    :param Q: current Q-values
    :param epsilon: exploration parameter
    :return: the chosen action
    """
    # get the available positions
    avail_indices, avail_mask = available(grid)

    if np.random.uniform(0, 1) < epsilon:
        # with probability epsilon make a random move (exploration)
        return avail_indices[np.random.randint(0, len(avail_indices))]
    else:
        # with probability 1-epsilon choose the action with the highest immediate reward (exploitation)
        q = Q[encode_state(grid)]
        q[np.logical_not(avail_mask)] = np.nan  # set the Q(state, action) with action currently non-available to nan
        max_indices = np.argwhere(q == np.nanmax(q))  # best action(s) along the available ones
        return int(max_indices[np.random.randint(0, len(max_indices))])  # ties are split randomly


class QPlayer:
    """
    Class to implement a player that plays according to the greedy policy defined
    by (empirical estimates of) the Q-values.
    """
    def __init__(self, Q, player='X'):
        """
        __init__
        :param self: self
        :param Q: Q-values
        :param player: 'X' or 'O'
        """
        self.Q = Q  # initialize Q-values
        self.player = player  # set the player

    def set_player(self, player='X', j=-1):
        """
        Set player to be either 'X' or 'O'
        :param self: self
        :param player: 'X' or 'O' ('X' by default)
        :param j: to change 'X' and 'O'
        """
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def act(self, grid, **kwargs):
        """
        Performs a greedy move, i.e. a (1-epsilon)-greedy action with epsilon equal to zero
        :param self: self
        :param grid: current state
        :param kwargs: keyword arguments
        :return: the action chosen greedily
        """
        return epsilon_greedy_action(grid, self.Q, 0)


def measure_performance(player_1, player_2, num_episodes=500):
    """
    Measures performance of player 1 against player 2
    :param player_1: first player (usually a QPlayer member)
    :param player_2: second player (usually a OptimalPlayer member)
    :param num_episodes: number of episodes played
    :return: percentage of wins of player_1 against player_2
    """
    meas = 0
    turns = np.array(['X', 'O'])
    env = TictactoeEnv()  # setting the environment
    for itr in range(num_episodes):
        env.reset()  # reset at the beginning of each episode
        grid, _, _ = env.observe()
        # alternating the turns to start
        player_1.set_player(turns[itr % 2])
        player_2.set_player(turns[(itr+1) % 2])
        while not env.end:
            if env.current_player == player_1.player:
                move = player_1.act(grid)  # move of the first player
            else:
                move = player_2.act(grid)  # move of the second player
            grid, _, _ = env.step(move, print_grid=False)  # updating the environment
        meas += env.reward(player=player_1.player)  # updating the reward of player_1
    return meas/num_episodes


def running_average(vec, windows_size=250):
    """
    Computes the running average of vec every windows_size elements
        Example: if windows_size=250 then it computes the mean of vec from 1 to 250, from 251 to 500 and so on.
    :param vec: numpy.ndarray
    :param windows_size: windows_size
    :return: mean of vec every windows_size elements
    """
    idx = np.arange(0, len(vec), windows_size)  # (i * windows_size for i = 1, ..., len(vec) / windows_size))
    return [np.sum(vec[i:i+windows_size])/windows_size for i in idx], idx + windows_size


def encode_state(state):
    """
    Construct Python bytes containing the raw data bytes in the array representing the state.
    :param state: numpy.ndarray
    :return: the bytes' representation of the state
    """
    return state.tobytes()


def available(grid):
    """
    Return available positions given a grid
    :param grid: current state
    :return:
        - avail_indices: list of available indices
        - avail_mask: list of bool which is True in available positions, False otherwise
    """
    avail_indices = []
    avail_mask = [False] * 9
    for i in range(9):
        pos = (int(i/3), i % 3)
        if grid[pos] == 0:
            avail_indices.append(i)  # add i to the available indices
            avail_mask[i] = True  # set the mask of the position i to True
    return avail_indices, avail_mask


def plot_stats(stats_dict, vec_var, var_name, var_legend_name, save=False, decaying_exploration=False):
    """
    Creates the plots for reward, M_opt, M_rand
    :param stats_dict: dictionary containing the training statistics
    :param vec_var: vector of variables to plot (need to be keys of stats_dict)
    :param var_name: variables name for the purpose of saving the plots
    :param var_legend_name: variables name for the legend (latex format)
    :param save: True to save figures in output folder
    :param decaying_exploration: True to plot the episode at which the decay stops and the exploration rate becomes constant
    """
    # creating the environment for the two plots
    fig_reward, ax_reward = plt.subplots()
    fig_performance, ax = plt.subplots(1, 2, figsize=(13.4, 4.8))
    fig_performance.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # adjust the spacing between subplots

    for var in vec_var:
        (stats, M_opt, M_rand) = stats_dict[var]
        # Plot of the average reward during training
        running_average_rewards, x_reward = running_average(stats['rewards'])  # compute average reward
        color = next(ax_reward._get_lines.prop_cycler)['color']
        ax_reward.plot(x_reward, running_average_rewards, label="$" + var_legend_name + " = " + str(var) + "$", color=color)
        if decaying_exploration:  # if exploration decay plot also the episode at which the decay stops
            find_nearest = np.abs(x_reward - 7/8 * var).argmin()  # from here constant 1-epsilon_min greedy policy
            if np.abs(x_reward - 7/8 * var).min() < 250:  # no plot if the nearest value is too far away (think of n_star = 40000)
                ax_reward.plot(x_reward[find_nearest], running_average_rewards[find_nearest], marker="o", color=color)
                ax_reward.vlines(x=x_reward[find_nearest], ymin=-0.8, ymax=0.8, color=color, ls='--')
        # Plot of M_opt and M_rand during training
        x_performance = np.arange(0, len(stats['rewards']) + 1, len(stats['rewards']) / (len(stats['test_Mopt']) - 1))
        ax[0].plot(x_performance, stats['test_Mopt'], label="$" + var_legend_name + " = " + str(var) + "$", color=color)
        ax[1].plot(x_performance, stats['test_Mrand'], label="$" + var_legend_name + " = " + str(var) + "$", color=color)
        print(var_name + " =", var, ": \tM_opt = ", M_opt, "\tM_rand = ", M_rand)  # print the performance

    ax_reward.set_ylim([-1, 1])
    ax_reward.set_xlabel('Episode')
    ax_reward.set_ylabel('Reward')
    ax_reward.set_title('Average reward during training')
    ax_reward.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     fancybox=True, shadow=True, ncol=4, fontsize=10)  # legend below outside the plot

    ax[0].hlines(y=0, xmin=x_reward[0], xmax=x_reward[-1],
                 color='r', linestyle='--')  # plot also the zero line for M_opt, as it is the highest M_opt achievable
    ax[0].set_ylim([-1, 0.1])
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('$M_{opt}$')
    ax[0].set_title('$M_{opt}$ during training')

    ax[1].set_ylim([-1, 1])
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('$M_{rand}$')
    ax[1].set_title('$M_{rand}$ during training')

    ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1.1, -0.15),
                            fancybox=True, shadow=True, ncol=5, fontsize=10)  # unique legend for the two plots
    plt.show()

    # saving onto file
    if save:
        output_folder = os.path.join(os.getcwd(), 'figures')  # set the output folder
        os.makedirs(output_folder, exist_ok=True)
        # saving figures in "png" format
        fig_performance.savefig(output_folder + '/performance_'+var_name+'.png', bbox_inches='tight')
        fig_reward.savefig(output_folder + '/rewards_'+var_name+'.png', bbox_inches='tight')
        # saving figures in "eps" format
        fig_performance.savefig(output_folder + '/performance_'+var_name+'.eps', format='eps', bbox_inches='tight')
        fig_reward.savefig(output_folder + '/rewards_'+var_name+'.eps', format='eps', bbox_inches='tight')


def plot_qtable(grid, Q, save=False, saving_name=None, show_legend=False):
    """
    Generates and saves a simil heatmap for the Q-values
    :param grid: current state
    :param Q: current Q-values
    :param save: True to save figures
    :param saving_name: saving name
    :param show_legend: True to show the legend
    :return:
    """
    text_lut = {0: np.nan, 1: 'X', -1: 'O'}
    q_vals = Q[encode_state(grid)][:]
    q_vals = q_vals.round(decimals=2)  # to avoid overlaps in the ggplot
    min_value = np.min(q_vals)  # get minimum value for the legend
    max_value = np.max(q_vals)  # get maximum value for the legend
    plot_data = pd.DataFrame({'x': np.tile([1, 2, 3], 3), 'y': np.repeat([1, 2, 3], 3),
                              'board_state': [text_lut[val] for val in grid.flatten()],
                              'Q': q_vals})  # creating the dataframe to be passed to ggplot
    plot = ggplot(plot_data, aes(x='x', y='y')) + \
        geom_tile(aes(fill='Q'), show_legend=show_legend) + \
        geom_text(aes(label='board_state'), color='white', size=30) + \
        geom_text(aes(label="Q"), show_legend=False) + \
        scale_fill_gradient2(limits=(min_value, max_value)) + \
        scale_y_reverse() + \
        theme(figure_size=(2, 2), axis_text=element_blank(), axis_ticks=element_blank(),
              strip_text_x=element_blank(), axis_title=element_blank())

# saving onto file
    if save:
        output_folder = os.path.join(os.getcwd(), 'figures/')  # set the output folder
        os.makedirs(output_folder, exist_ok=True)
        fname = output_folder + saving_name
        plot.save(filename=fname + '.png')
        plot.save(filename=fname + '.eps')


def heatmaps_subplots(grids, Q):
    """
    Generate heatmaps for all states in grids
    :param grids: current states
    :param Q: current Q-values
    :return:
    """
    for (num, grid) in enumerate(grids):
        grid = np.array(grid)
        name = 'heatmap_' + str(num)
        if num == len(grids) - 1:
            plot_qtable(grid, Q, save=True, saving_name=name, show_legend=True)
        else:
            plot_qtable(grid, Q, save=True, saving_name=name)

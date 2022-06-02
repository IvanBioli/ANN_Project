import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import time
import pickle
from tqdm import tqdm
from plotnine import *
import os


"COMMON UTILS FUNCTION FOR TABULAR AND DEEP Q LEARNING"


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
            try:
                grid, _, _ = env.step(move, print_grid=False)  # updating the environment
            except ValueError:
                env.end = True  # the current player chose an unavailable action, thus reward = -1
                if env.current_player == 'X':
                    env.winner = 'O'
                else:
                    env.winner = 'X'
        meas += env.reward(player=player_1.player)  # updating the reward of player_1
    return meas/num_episodes


def running_average(vec, windows_size=250, no_idx=False):
    """
    Computes the running average of vec every windows_size elements
        Example: if windows_size=250 then it computes the mean of vec from 1 to 250, from 251 to 500 and so on
    :param vec: numpy.ndarray
    :param windows_size: windows_size
    :param no_idx: True not to return the indices of vec at which the mean is computed
    :return:
        - mean of vec every windows_size elements
        - indices of vec at which the mean is computed
    """
    idx = np.arange(0, len(vec), windows_size)  # (i * windows_size for i = 1, ..., len(vec) / windows_size))
    if no_idx:
        return np.array([np.sum(vec[i:i + windows_size]) / windows_size for i in idx])
    else:
        return np.array([np.sum(vec[i:i+windows_size])/windows_size for i in idx]), idx + windows_size + 1


def encode_state(state):
    """
    Construct Python bytes containing the raw data bytes in the array representing the state.
    :param state: numpy.ndarray
    :return:
        - the bytes' representation of the state
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


def return_lambda_explor(epsilon_min, epsilon_max, n_star):
    """
    Rule for the decay of the exploration rate during training
    :param epsilon_min: minimum allowed exploration rate
    :param epsilon_max: maximum allowed exploration rate
    :param n_star: parameter which governs the decay rate of epsilon
    :return:
        - the exploration rule at training episode n
    """
    return lambda n: np.max([epsilon_min, epsilon_max * (1 - n/n_star)])


def stats_averaging(stats_dict_list, windows_size=250):
    """
    Function that performs averaging of the stats collected during many training runs
    :param stats_dict_list: a list of dictionaries with all the stats collected
    :param windows_size: window size for the testing of the performance of the agent
    :return:
        - stats_dict_avg: a dictionary with the average over the training runs of all quantities of interest

    Example:
    stats_dict_list in an object of the form [dict_1, ..., dict_N]
    Each dict_i is a dictionary with all the stats from the first run for all parameters,
    so it is an object of the form {'var_values_1': (stats_1, M_opt_1, M_rand_1), 'var_values_2':
    (stats_2, M_opt_2, M_rand_2), ..., ...}.
    Recall that stats_i are themselves dictionaries of the form
    {'rewards': list of the rewards during training, 'test_Mopt': list of the test values for M_opt during training,
    'test_Mrand': list of the test values for M_rand during training}
    """
    # initialize the final dictionary with the keys taken from the first run (they are always the same)
    stats_dict_avg = dict.fromkeys(stats_dict_list[0].keys(), list())
    for var in stats_dict_list[0].keys():  # for loop over all parameters
        stats = {}  # initialize the dictionary with the stats
        # compute means of final measurements of M_opt and M_rand
        M_opt = np.median([stats_dict[var][1] for stats_dict in stats_dict_list], axis=0)
        M_rand = np.median([stats_dict[var][2] for stats_dict in stats_dict_list], axis=0)
        # get the training stats for the current parameter
        (tmp_stats, _, _) = stats_dict_list[0][var]
        for key in tmp_stats.keys():  # for loop over all the keys of the current dictionary (stats_i in the example)
            if key == 'rewards' or key == 'loss':
                # for the rewards, compute the mean by calling the running average performance on each dictionary
                stats[key] = np.median([running_average(stats_dict[var][0][key],
                                                        windows_size=windows_size, no_idx=True)
                                        for stats_dict in stats_dict_list], axis=0)
                # compute the 25 and 75 percentiles of the values obtained in the training runs
                stats[key + '_25'] = np.percentile([running_average(stats_dict[var][0][key],
                                                                    windows_size=windows_size, no_idx=True)
                                                    for stats_dict in stats_dict_list], q=25, axis=0)
                stats[key + '_75'] = np.percentile([running_average(stats_dict[var][0][key],
                                                                    windows_size=windows_size, no_idx=True)
                                                    for stats_dict in stats_dict_list], q=75, axis=0)
                if key == 'loss':
                    stats['loss'] = stats['loss'][1:]
                    stats['loss_25'] = stats['loss_25'][1:]
                    stats['loss_75'] = stats['loss_75'][1:]
            else:
                # compute the mean directly from the dictionary
                stats[key] = np.median([stats_dict[var][0][key] for stats_dict in stats_dict_list], axis=0)
                # compute the percentiles' deviation directly from the dictionary
                stats[key + '_25'] = np.percentile([stats_dict[var][0][key] for stats_dict in stats_dict_list],
                                                   q=25, axis=0)
                stats[key+'_75'] = np.percentile([stats_dict[var][0][key] for stats_dict in stats_dict_list],
                                                 q=75, axis=0)
        # Saving in the stats_dict_avg
        stats_dict_avg[var] = (stats, M_opt, M_rand)

    return stats_dict_avg


def plot_stats(stats_dict_list, vec_var, var_name, var_legend_name, save=False, decaying_exploration=False,
               perc=False, windows_size=250, keys=None):
    """
    Creates the plots for reward, M_opt, M_rand
    :param keys: keys of the variables to be plotted
    :param stats_dict_list: list of dictionaries containing the training statistics
    :param vec_var: vector of variables to plot (need to be keys of stats_dict)
    :param var_name: variables name for the purpose of saving the plots
    :param var_legend_name: variables name for the legend (latex format)
    :param save: True to save figures in output folder
    :param decaying_exploration: True to plot the episode at which the decay stops
        and the exploration rate becomes constant
    :param perc: True to show the 25 and 75 percentiles of the different measurements from many training runs
    :param windows_size: windows size for averaging rewards
    """
    # averaging over multiple training runs
    stats_dict = stats_averaging(stats_dict_list, windows_size=windows_size)
    # defining variables to plot
    (stats, M_opt, M_rand) = stats_dict[vec_var[0]]  # reference element from the dictionary
    if keys is None:
        keys = stats.keys()
    # creating the environment for the two plots
    if 'loss' in keys and 'rewards' in keys:
        fig_1, ax_1 = plt.subplots(1, 2, figsize=(13.4, 4.8), squeeze=False)
        fig_1.tight_layout(pad = 7.)
        fig_1.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # adjust the spacing between subplots
    elif 'loss' in keys or 'rewards' in keys:
        fig_1, ax_1 = plt.subplots(1, 1, squeeze=False)
    if 'test_Mopt' in keys:
        fig_performance, ax = plt.subplots(1, 2, figsize=(13.4, 4.8))
        fig_performance.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # adjust spacing between subplots

    for var in vec_var:
        (stats, M_opt, M_rand) = stats_dict[var]
        for key in keys:
            # Plot of the average reward/loss during training
            if key == 'rewards' or key == 'loss':
                idx = int(key == 'loss')  # idx = 0 for reward and idx = 1 for loss
                running_average = stats[key]
                if key == 'rewards':
                    x_1 = np.arange(0, len(running_average)*windows_size, windows_size)
                else:
                    x_1 = np.arange(windows_size, (len(running_average)+1) * windows_size, windows_size)
                color = next(ax_1[0, idx]._get_lines.prop_cycler)['color']
                ax_1[0, idx].plot(x_1, running_average, label="$" +
                                                              var_legend_name + " = " + str(var) + "$", color=color)
                if perc:
                    ax_1[0, idx].fill_between(x_1, stats[key+'_25'], stats[key+'_75'], alpha=0.2)
                if decaying_exploration:  # if exploration decay plot also the episode at which the decay stops
                    find_nearest = np.abs(x_1 - 7/8 * var).argmin()  # from here constant 1-epsilon_min greedy policy
                    if np.abs(x_1 - 7/8 * var).min() < windows_size:
                        # no plot if the nearest value is too far away (think of n_star = 40000)
                        ax_1[0, idx].plot(x_1[find_nearest], running_average[find_nearest], marker="o", color=color)
                        ax_1[0, idx].vlines(x=x_1[find_nearest], ymin=min(running_average), ymax=max(running_average),
                                            color=color, ls='--')
                # Legend and axis names
                if 'loss' in keys and 'rewards' in keys:
                    ax_1[0, idx].set_xlabel('Episode', fontsize='x-large')
                    ax_1[0, idx].set_ylabel(key.capitalize(), fontsize='xx-large')
                    ax_1[0, idx].set_title('Average ' + key + ' during training', fontsize='xx-large')
                    ax_1[0, idx].tick_params(axis='x', labelsize='x-large')
                    ax_1[0, idx].tick_params(axis='y', labelsize='x-large')
                    ax_1.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1.1, -0.15), fancybox=True,
                                              shadow=True, ncol=5, fontsize='xx-large')  # unique legend for the two plots
                else:
                    ax_1[0, idx].set_xlabel('Episode')
                    ax_1[0, idx].set_ylabel(key.capitalize())
                    ax_1[0, idx].set_title('Average ' + key + ' during training')
                    ax_1[0, idx].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                                        fancybox=True, shadow=True, ncol=4)  # legend outside the plot

            # Plot of M_opt and M_rand during training
            if key == 'test_Mopt' or key == 'test_Mrand':
                idx = int(key == 'test_Mrand')   # idx = 0 for Mopt and idx = 1 for Mrand
                x_performance = np.arange(0, len(stats['rewards']) * windows_size + 1,
                                          len(stats['rewards']) * windows_size / (len(stats[key]) - 1))
                color = next(ax[idx]._get_lines.prop_cycler)['color']
                ax[idx].plot(x_performance, stats[key], label="$" + var_legend_name + " = "
                                                              + str(var) + "$", color=color)
                if decaying_exploration:
                    find_nearest = np.abs(x_performance-7/8 * var).argmin()
                    if np.abs(x_performance - 7/8 * var).min() < windows_size:
                        ax[idx].plot(x_performance[find_nearest], stats[key][find_nearest], marker="o", color=color)
                        ax[idx].vlines(x=x_performance[find_nearest], ymin=min(stats[key]), ymax=max(stats[key]),
                                       color=color, ls='--')
                if perc:
                    ax[idx].fill_between(x_performance, stats[key+'_25'], stats[key+'_75'], alpha=0.2)
                # Legend and axis names
                ax[0].hlines(y=0, xmin=x_performance[0], xmax=x_performance[-1],
                             color='r', linestyle='--')  # plot also the zero line for M_opt (highest M_opt achievable)
                ax[0].set_ylim([-1, 0.1])
                ax[0].set_xlabel('Episode',fontsize='x-large')
                ax[0].set_ylabel('$M_{opt}$',fontsize='xx-large')
                ax[0].set_title('$M_{opt}$ during training',fontsize='xx-large')
                ax[0].locator_params(axis='x', nbins=5)
                ax[0].tick_params(axis='x', labelsize='x-large')
                ax[0].tick_params(axis='y', labelsize='x-large')
                # ax[1].set_ylim(None, 1)
                ax[1].set_xlabel('Episode',fontsize='x-large')
                ax[1].set_ylabel('$M_{rand}$',fontsize='xx-large')
                ax[1].set_title('$M_{rand}$ during training',fontsize='xx-large')
                ax[1].locator_params(axis='x', nbins=5)
                ax[1].tick_params(axis='x', labelsize='x-large')
                ax[1].tick_params(axis='y', labelsize='x-large')
                ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1.1, -0.15), fancybox=True, shadow=True,
                                        ncol=5, fontsize='xx-large')  # unique legend for the two plots

        print(var_name + " =", var, ": \tM_opt = ", M_opt, "\tM_rand = ", M_rand)  # print the performance
    plt.show()

    # saving onto file
    if save:
        output_folder = os.path.join(os.getcwd(), 'figures')  # set the output folder
        os.makedirs(output_folder, exist_ok=True)
        # saving figures in "png" and "pdf" format
        if 'loss' in keys or 'rewards' in keys:
            fig_1.savefig(output_folder + '/rewards_'+var_name+'.png', bbox_inches='tight')
            fig_1.savefig(output_folder + '/rewards_' + var_name + '.pdf', format='pdf', bbox_inches='tight')
        if 'test_Mopt' in keys:
            fig_performance.savefig(output_folder + '/performance_' + var_name + '.png', bbox_inches='tight')
            fig_performance.savefig(output_folder + '/performance_' + var_name + '.pdf', format='pdf',
                                    bbox_inches='tight')


"UTILS FUNCTIONS FOR TABULAR Q LEARNING"


def epsilon_greedy_action(grid, Q, epsilon):
    """
    Performs an epsilon-greedy action starting from a given state and given Q-values
    :param grid: current state
    :param Q: current Q-values
    :param epsilon: exploration parameter
    :return:
        - the chosen action
    """
    # get the available positions
    avail_indices, avail_mask = available(grid)

    if np.random.uniform(0, 1) < epsilon:
        # with probability epsilon make a random move (exploration)
        return int(np.random.choice(avail_indices))
    else:
        # with probability 1-epsilon choose the action with the highest immediate reward (exploitation)
        q = np.copy(Q[encode_state(grid)])
        q[np.logical_not(avail_mask)] = np.nan  # set the Q(state, action) with action currently non-available to nan
        max_indices = np.argwhere(q == np.nanmax(q)).flatten()  # best action(s) along the available ones
        return int(np.random.choice(max_indices))  # ties are split randomly


def plot_qtable(grid, Q, save=False, saving_name=None, show_legend=False):
    """
    Generates and saves a heatmap for the Q-values
    :param grid: current state
    :param Q: current Q-values
    :param save: True to save figures
    :param saving_name: saving name
    :param show_legend: True to show the legend
    :return:
    """
    text_lut = {0: np.nan, 1: 'X', -1: 'O'}
    avail_indices, avail_mask = available(grid)
    q_vals = np.copy(Q[encode_state(grid)]).round(decimals=2)
    q_vals[np.logical_not(avail_mask)] = np.nan  # for unavailable actions
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

    print(plot)  # show the plot

    # saving onto file
    if save:
        output_folder = os.path.join(os.getcwd(), 'figures/')  # set the output folder
        os.makedirs(output_folder, exist_ok=True)
        fname = output_folder + saving_name
        plot.save(filename=fname + '.png', verbose=False)
        plot.save(filename=fname + '.pdf', verbose=False)


def heatmaps_subplots(grids, Q, save):
    """
    Generate heatmaps for all states in grids
    :param save: True to save the figure
    :param grids: current states
    :param Q: current Q-values
    :return:
    """
    for (num, grid) in enumerate(grids):
        grid = np.array(grid)
        name = 'heatmap_' + str(num)
        plot_qtable(grid, Q, save=save, saving_name=name)


"UTILS FUNCTIONS FOR DEEP Q LEARNING"


# Network defined in the project description
def create_q_model():
    """
    Creates a simple MLP with 2 hidden layers of 128 neurons each which takes as input the state representation
    The activation function at the hidden layers is ReLU while there is linear activation at the output layer
    :return:
        - the MLP model
    """
    hidden_neurons = 128
    num_actions = 9
    # Inputs of shape = (3, 3, 2)
    inputs = layers.Input(shape=(3, 3, 2,))

    layer0 = layers.Flatten()(inputs)  # flattening
    # Two fully connected hidden layers each with 128 neurons and ReLU activation
    layer1 = layers.Dense(units=hidden_neurons, activation="relu")(layer0)
    layer2 = layers.Dense(units=hidden_neurons, activation="relu")(layer1)
    # Output with linear activation function
    action = layers.Dense(num_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)


def grid_to_tensor(grid, player):
    """
    Converts a grid of 'X' and 'O' into the desired input representation of the neural network
    Creates a tensor of shape (3, 3) for player='X' or 'O' with 1's where the player has played, 0's otherwise.
    The combination of such tensors for two players is given as input to the network
    :param grid: grid
    :param player: the player
    :return:
    """
    if player == 'X':
        return tf.convert_to_tensor(np.stack((np.where(grid == 1, 1, 0), np.where(grid == -1, 1, 0)), -1))
    else:
        return tf.convert_to_tensor(np.stack((np.where(grid == -1, 1, 0), np.where(grid == 1, 1, 0)), -1))


def dqn_epsilon_greedy(model, state_tensor, epsilon):
    """
    Chooses an epsilon-greedy action according to the current model (i.e. estimated Q-values)
    :param model: the current model
    :param state_tensor: tensor representation of the current state, compatible with the desired network input
    :param epsilon: exploration rate
    :return:
        - the chosen action
    """
    # Use epsilon-greedy for exploration
    if epsilon > np.random.uniform(0, 1):
        # Take random action
        avail_indices = np.argwhere(state_tensor[0, :, :, 0].numpy().flatten() ==
                                    state_tensor[0, :, :, 1].numpy().flatten()).flatten()
        return int(np.random.choice(avail_indices))
    else:
        # Predict action Q-values
        # From environment state
        action_probs = model(state_tensor, training=False).numpy().flatten()
        # Take best action
        q_max = np.amax(action_probs)
        max_indices = np.argwhere(action_probs == q_max).flatten()
        return int(np.random.choice(max_indices))  # ties are split randomly


def plot_deep_qtable(grid, model, save=False, saving_name=None, show_legend=False):
    """
    Generates and saves a simil heatmap for the Q-values
    :param model: the current model
    :param grid: current state
    :param save: True to save figures
    :param saving_name: saving name
    :param show_legend: True to show the legend
    :return:
    """
    text_lut = {0: np.nan, 1: 'X', -1: 'O'}
    num_X = np.count_nonzero(grid == 1.)
    num_O = np.count_nonzero(grid == -1.)
    if num_X > num_O:
        player = 'O'
    else:
        player = 'X'
    tensor_grid = grid_to_tensor(grid, player)
    tensor_grid = tf.expand_dims(tensor_grid, axis=0)
    q_vals = np.round(model.predict(tensor_grid)[0].astype('float64'), decimals=2)  # compute output of the network
    avail_indices, avail_mask = available(grid)
    q_vals[np.logical_not(avail_mask)] = np.nan  # for unavailable actions
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

    print(plot)  # show the plot

    # saving onto file
    if save:
        output_folder = os.path.join(os.getcwd(), 'figures/')  # set the output folder
        os.makedirs(output_folder, exist_ok=True)
        fname = output_folder + saving_name
        plot.save(filename=fname + '.png', verbose=False)
        plot.save(filename=fname + '.pdf', verbose=False)


def heatmaps_deep_subplots(grids, model, save):
    """
    Generate heatmaps for all states in grids
    :param save: True to save the figures
    :param model: the current model
    :param grids: current states
    :return:
    """
    for (num, grid) in enumerate(grids):
        grid = np.array(grid)
        name = 'deep_heatmap_' + str(num)
        plot_deep_qtable(grid, model, save=save, saving_name=name)

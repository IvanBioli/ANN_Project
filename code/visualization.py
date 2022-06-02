import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
from utils import *
from q_learning import *
from deep_q_learning import *
from train_multiple_runs import *


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


"PLOT FUNCTIONS FOR TABULAR Q LEARNING"
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


"PLOT FUNCTIONS FOR DEEP Q LEARNING"

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

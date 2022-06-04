import time
import pickle
import os
from q_learning import *
from deep_q_learning import *
from utils import *


def train_avg(var_name, var_values, q_learning_params_list, dqn=False, num_avg=10, save_stats=True):
    """
    Function that computes all the quantities of interest averaging over many training runs
    :param dqn:
    :param var_name: name of the parameter
    :param var_values: values for the parameter var_name
    :param q_learning_params_list: list of dictionaries with the parameters
        for the Q-learning for each value of var_name, for example {'test_freq': 250, 'self_practice': True, ...}
    :param num_avg: number of training runs
    :param save_stats: True to save the stats
    :return:
        - stats_dict_list: list of dictionaries which contain all the stats for the different values of var_name
    """
    stats_dict_list = []
    for i in range(num_avg):
        print('************** RUN', i+1, 'OF', num_avg, '**************')
        stats_dict = {}  # initialize the dictionary for the current training run
        for (idx, var) in enumerate(var_values):
            print("------------- Training with " + var_name + " =", var, "-------------")
            start = time.time()
            # get the dictionary of the current parameters for the Q-learning
            q_learning_params = q_learning_params_list[idx]
            if not dqn:
                # train a Q-learning agent with the current parameters
                Q, stats = q_learning(**q_learning_params)
                # measure the final performance
                M_opt = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=0.))
                M_rand = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=1.))
            else:
                # train a Deep Q-Learning agent with the current parameters
                model, stats = deep_q_learning(**q_learning_params)
                # measure the final performance
                M_opt = measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=0.))
                M_rand = measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=1.))
            print("M_opt =", M_opt)
            print("M_rand =", M_rand)
            # insert the stats of the current parameter in the dictionary of the current run
            stats_dict.update({var: (stats, M_opt, M_rand)})
            elapsed = time.time() - start
            print("Training with " + var_name + " =", var, " took:",
                  time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed)), "\n\n")
        # append the dictionary of the current run to the overall list
        stats_dict_list.append(stats_dict)

        # saving onto file
        if save_stats:
            output_folder = os.path.join(os.getcwd(), 'results')  # set the folder
            os.makedirs(output_folder, exist_ok=True)
            fname = output_folder + '/stats_dict_' + var_name + '_list.pkl'
            with open(fname, 'wb') as handle:
                pickle.dump(stats_dict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stats_dict_list


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

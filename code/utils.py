from tic_env import *


def measure_performance(player_1, player_2, num_episodes=500):
    """
    Measures performance of player 1 against player 2
    :param player_1: first player (usually a QPlayer member)
    :param player_2: second player (usually a OptimalPlayer member)
    :param num_episodes: number of episodes played
    :return:
        - percentage of wins of player_1 against player_2
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


def compute_training_time(stats_dict, optimal_params, num_avg, percentage=0.8, test_freq=250, quantiles=False):
    """
    Compute T_train as the maximum between the values obtained for M_opt and M_rand
    :param stats_dict: dictionary containing all the results of interest, including those of optimal_params
    :param optimal_params: the optimal parameters
    :param num_avg: number of training runs on which to average
    :param percentage: percentage at which T_train is calculated
    :param quantiles: True to print the quantiles
    :param test_freq: to round the number of episodes defining T_train
    :return:
        - print median (and possibly quantiles) of T_train for all the training runs
    """

    stats_dict_best_param = [stats_dict[i][optimal_params] for i in range(num_avg)]
    final_m_opt = [stats_dict_best_param[i][1] for i in range(num_avg)]
    test_m_opt = [stats_dict_best_param[i][0]['test_Mopt'] for i in range(num_avg)]
    final_m_rand = [stats_dict_best_param[i][2] for i in range(num_avg)]
    test_m_rand = [stats_dict_best_param[i][0]['test_Mrand'] for i in range(num_avg)]

    # Compute training time
    starting_m_opt = [test_m_opt[i][0] for i in range(num_avg)]
    train_times_m_opt = np.array([np.where(np.array(test_m_opt[i]) > starting_m_opt[i]
                                           + percentage * (final_m_opt[i]-starting_m_opt[i]))[0][0]
                                  for i in range(num_avg)])
    train_times_m_opt = train_times_m_opt * test_freq
    starting_m_rand = [test_m_rand[i][0] for i in range(num_avg)]
    train_times_m_rand = np.array([np.where(np.array(test_m_rand[i]) > starting_m_rand[i]
                                            + percentage * (final_m_rand[i]-starting_m_rand[i]))[0][0]
                                   for i in range(num_avg)])
    train_times_m_rand = train_times_m_rand * test_freq
    train_times = [np.maximum(train_times_m_opt[i], train_times_m_rand[i]) for i in range(num_avg)]

    print("Median:\t M_opt = ", np.round(np.median(final_m_opt), decimals=2),
              "\t M_rand = ", np.round(np.median(final_m_rand), decimals=2),
              "\t T_train = ", np.round(np.median(train_times), decimals=2))
    if quantiles:
        print("25th quantile:\t M_opt = ", np.round(np.percentile(final_m_opt, q=25), decimals=2),
              "\t M_rand = ", np.round(np.percentile(final_m_rand, q=25), decimals=2),
              "\t T_train = ", np.round(np.percentile(train_times, q=25), decimals=2))
        print("75th quantile:\t M_opt = ", np.round(np.percentile(final_m_opt, q=75), decimals=2),
              "\t M_rand = ", np.round(np.percentile(final_m_rand, q=75), decimals=2),
              "\t T_train = ", np.round(np.percentile(train_times, q=75), decimals=2))

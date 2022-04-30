from utils import *
from collections import defaultdict
import time


def q_learning_against_opt(env, alpha=0.5, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                           epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False):
    """
    Trains a Q-Learning agent by playing against optimal strategy (up to an epsilon_opt level of randomness,
    i.e. in expectation the teacher plays the optimal move epsilon_opt % of the time)
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param epsilon_exploration: exploration rate
        If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
        is taken with probability (1-epsilon_exploration)
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the action
        with the highest Q-value is taken with probability (1-epsilon_exploration_rule(n))
    :param epsilon_opt: float, in [0, 1]. Probability of the teacher making a random action instead
        of the optimal action at any given time.
    :param test_freq: frequency (in number of episodes) of tests executed against optimal and fully random policies
    :param verbose: bool
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training
    """
    turns = np.array(['X', 'O'])
    # Q-values map
    # Dictionary that maps the np.ndarray.tobyte() representation of the grid to an array of action values
    Q = defaultdict(lambda: np.zeros(9))    # All Q-values are initialized to 0
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    if test_freq is not None:
        episode_Mopt = [measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=0.))]
        episode_Mrand = [measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=1.))]
    else:
        episode_Mopt = []  # initialize
        episode_Mrand = []  # initialize
    if verbose and (test_freq is not None):
        print('Episode  0 :\tM_opt = ', episode_Mopt[0], '\tM_rand = ', episode_Mrand[0])
    # Rule for exploration
    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    for itr in range(num_episodes):
        my_player = turns[itr % 2]
        player_opt = OptimalPlayer(epsilon=epsilon_opt, player=turns[(itr+1) % 2])
        env.reset()
        state, _, _ = env.observe()
        # First step of the adversarial
        if env.current_player == player_opt.player:
            move = player_opt.act(state)
            state, _, _ = env.step(move)
        action = epsilon_greedy_action(state, Q, epsilon_exploration_rule(itr+1))
        while not env.end:
            next_state, _, _ = env.step(action)     # Move according to the policy
            if not env.end:
                move = player_opt.act(next_state)   # Adversarial move
                next_state, _, _ = env.step(move)
            # Sarsa update rule
            reward = env.reward(player=my_player)
            if not env.end:
                next_action = epsilon_greedy_action(next_state, Q, epsilon_exploration_rule(itr+1))
                target = reward + gamma * Q[encode_state(next_state)][next_action]
            else:
                target = reward
            Q[encode_state(state)][action] += alpha * (target - Q[encode_state(state)][action])
            # Preparing for the next move
            state = next_state
            action = next_action

        episode_rewards[itr] = env.reward(player=my_player)

        # Testing the performance
        if (test_freq is not None) and ((itr+1) % test_freq == 0):
            M_opt = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=0.))
            M_rand = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=1.))
            episode_Mopt.append(M_opt)
            episode_Mrand.append(M_rand)
            if verbose:
                print('Episode ', itr+1, ':\tM_opt = ', M_opt, '\tM_rand = ', M_rand)

    # Dictionary of stats
    stats = {
        'rewards': episode_rewards,
        'test_Mopt': episode_Mopt,
        'test_Mrand': episode_Mrand,
    }
    return Q, stats


def q_learning_self_practice(env, alpha=0.5, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                             epsilon_exploration_rule=None, test_freq=None, verbose=False):
    """
    Trains a Q-Learning agent by self-practice (i.e. plays against itself).
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param epsilon_exploration: exploration rate
        If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
        is taken with probability (1-epsilon_exploration)
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the action
        with the highest Q-value is taken with probability (1-epsilon_exploration_rule(n))
    :param test_freq: frequency (in number of episodes) of tests executed against optimal and fully random policies
    :param verbose: bool
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training
    """
    turns = np.array(['X', 'O'])
    # Q-values map
    # Dictionary that maps the np.ndarray.tobyte() representation of the grid to an array of action values
    Q = defaultdict(lambda: np.zeros(9))    # All Q-values are initialized to 0
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    # Stats of training
    if test_freq is not None:
        episode_Mopt = [measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=0.))]
        episode_Mrand = [measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=1.))]
    else:
        episode_Mopt = []  # initialize
        episode_Mrand = []  # initialize
    if verbose and (test_freq is not None):
        print('Episode  0 :\tM_opt = ', episode_Mopt[0], '\tM_rand = ', episode_Mrand[0])
    # Rule for exploration
    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if no exploration rule is given, it is the constant one

    for itr in range(num_episodes):
        my_player = turns[itr % 2]
        env.reset()
        state, _, _ = env.observe()
        # First two turns outside the loop (at least five turns are played)
        action = epsilon_greedy_action(state, Q, epsilon_exploration_rule(itr + 1))
        state_adv, _, _ = env.step(action)
        action_adv = epsilon_greedy_action(state_adv, Q, epsilon_exploration_rule(itr + 1))
        while not env.end:
            # Adversarial turn
            state_adv, _, _ = env.observe()
            reward = - env.reward(player=env.current_player)  # Reward of the player who made the last move
            next_state, _, _ = env.step(action_adv)
            # Player's turn
            if not env.end:
                next_action = epsilon_greedy_action(next_state, Q, epsilon_exploration_rule(itr + 1))
                target = reward + gamma * Q[encode_state(next_state)][next_action]
            else:   # action_adv is the one that makes the game end
                reward = - env.reward(player=env.current_player)    # reward of the player who made the game end, i.e. the adversary of the current player
                # Update for the adversary of the current player
                Q[encode_state(state_adv)][action_adv] += alpha * (reward - Q[encode_state(state_adv)][action_adv])
                # Target for the current player
                target = - reward
            Q[encode_state(state)][action] += alpha * (target - Q[encode_state(state)][action])

            # Preparing for the next iteration
            action = action_adv
            state = state_adv
            action_adv = next_action

        episode_rewards[itr] = env.reward(player=my_player)

        # Testing the performance
        if (test_freq is not None) and ((itr+1) % test_freq == 0):
            M_opt = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=0.))
            M_rand = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=1.))
            episode_Mopt.append(M_opt)
            episode_Mrand.append(M_rand)
            if verbose:
                print('Episode ', itr+1, ':\tM_opt = ', M_opt, '\tM_rand = ', M_rand)

    # Dictionary of stats
    stats = {
        'rewards': episode_rewards,
        'test_Mopt': episode_Mopt,
        'test_Mrand': episode_Mrand,
    }
    return Q, stats


def q_learning(env, alpha=0.5, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
               epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False,
               against_opt=False, self_practice=False):
    """
    Calls q_learning_against_opt if against_opt = True and q_learning_self_practice if self_practice = True
    """
    if int(against_opt) + int(self_practice) != 1:
        raise ValueError("Please choose a training method")
    if against_opt:
        return q_learning_against_opt(env, alpha, gamma, num_episodes, epsilon_exploration,
                                      epsilon_exploration_rule,  epsilon_opt, test_freq, verbose)
    else:
        return q_learning_self_practice(env, alpha, gamma, num_episodes, epsilon_exploration,
                                        epsilon_exploration_rule, test_freq, verbose)


def train_avg(var_name, var_values, q_learning_params_list, num_avg=10, save_stats=True):
    """
    Function that computes all the quantities of interest by averaging over many training runs
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
            # perform Q-learning with the current parameters
            Q, stats = q_learning(**q_learning_params)
            # measure the final performance
            M_opt = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=0.))
            M_rand = measure_performance(QPlayer(Q=Q), OptimalPlayer(epsilon=1.))
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

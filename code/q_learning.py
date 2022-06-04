from tic_env import OptimalPlayer
from collections import defaultdict
from utils import *


def encode_state(state):
    """
    Construct Python bytes containing the raw data bytes in the array representing the state.
    :param state: numpy.ndarray
    :return:
        - the bytes' representation of the state
    """
    return state.tobytes()


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

    def set_player(self, player='X'):
        """
        Set player to be either 'X' or 'O'
        :param self: self
        :param player: 'X' or 'O' ('X' by default)
        """
        self.player = player

    def act(self, grid, **kwargs):
        """
        Performs a greedy move, i.e. a (1-epsilon)-greedy action with epsilon equal to zero
        :param self: self
        :param grid: current state
        :param kwargs: keyword arguments
        :return: the action chosen greedily
        """
        return epsilon_greedy_action(grid, self.Q, 0)


def q_learning_against_opt(env, alpha=0.05, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
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
        of the optimal action at any given time
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
            # Q-Learning update rule
            reward = env.reward(player=my_player)
            if not env.end:
                next_action = epsilon_greedy_action(next_state, Q, epsilon_exploration_rule(itr+1))
                next_greedy_action = epsilon_greedy_action(next_state, Q, 0)
                target = reward + gamma * Q[encode_state(next_state)][next_greedy_action]
            else:
                target = reward  # the fictitious Q-value of Q(next_state)[\cdot] is zero
            Q[encode_state(state)][action] += alpha * (target - Q[encode_state(state)][action])  # update Q-value
            # Preparing for the next move
            state = next_state
            action = next_action

        episode_rewards[itr] = env.reward(player=my_player)  # reward of the current episode

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


def q_learning_self_practice(env, alpha=0.05, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
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
        my_player = turns[itr % 2]  # switch every game the first turn
        env.reset()
        # First two turns outside the while loop (at least five turns are played)
        state, _, _ = env.observe()  # agent's first state
        action = epsilon_greedy_action(state, Q, epsilon_exploration_rule(itr+1))  # agent's first action
        state_adv, _, _ = env.step(action)  # the adversary first state
        action_adv = epsilon_greedy_action(state_adv, Q, epsilon_exploration_rule(itr + 1))  # adversary act
        while not env.end:
            state_adv, _, _ = env.observe()
            next_state, _, _ = env.step(action_adv)
            reward = env.reward(player=env.current_player)  # reward of the current player
            if not env.end:
                next_action = epsilon_greedy_action(next_state, Q, epsilon_exploration_rule(itr+1))
                next_greedy_action = epsilon_greedy_action(next_state, Q, 0)
                target = reward + gamma * Q[encode_state(next_state)][next_greedy_action]  # set target
            else:
                target = reward  # fictitious Q-value equal to zero for next_state
                # adversarial update
                Q[encode_state(state_adv)][action_adv] += alpha * (-reward - Q[encode_state(state_adv)][action_adv])

            Q[encode_state(state)][action] += alpha * (target - Q[encode_state(state)][action])  # update

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


def q_learning(env, alpha=0.05, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
               epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False,
               against_opt=False, self_practice=False):
    """
    Calls q_learning_against_opt if against_opt = True and q_learning_self_practice if self_practice = True
    """
    if int(against_opt) + int(self_practice) != 1:
        raise ValueError("Please, choose a training method")
    if against_opt:
        return q_learning_against_opt(env, alpha, gamma, num_episodes, epsilon_exploration,
                                      epsilon_exploration_rule,  epsilon_opt, test_freq, verbose)
    else:
        return q_learning_self_practice(env, alpha, gamma, num_episodes, epsilon_exploration,
                                        epsilon_exploration_rule, test_freq, verbose)

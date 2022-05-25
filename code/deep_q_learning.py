from utils import *


class DeepQPlayer:
    """
    Class to implement a player that plays according to the greedy policy defined
    by (empirical estimates of) the Q-values as the output of an MLP.
    """
    def __init__(self, model, player='X'):
        """
        __init__
        :param self: self
        :param model:
        :param player: 'X' or 'O'
        """
        self.model = model  # initialize model
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
        Performs a greedy move
        :param self: self
        :param grid: current state
        :param kwargs: keyword arguments
        :return: the action chosen greedily
        """
        grid = tf.expand_dims(grid_to_tensor(grid, self.player), axis=0)
        action_probs = self.model(grid, training=False).numpy().flatten()
        # Take best action
        max_indices = np.argwhere(action_probs == np.amax(action_probs)).flatten()
        return int(np.random.choice(max_indices))  # ties are split randomly


def deep_q_learning_against_opt(env, lr=5e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                                epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False,
                                batch_size=64, max_memory_length=10000, update_target_network=500, update_freq=1):
    """
    This function trains a Deep Q-agent to play Tic-Tac-Toe against the optimal strategy
    :param update_freq: update frequency (frames)
    :param update_target_network: how often to update the target network (num games)
    :param max_memory_length: replay buffer size
    :param batch_size: batch size for the sampling when updating
    :param lr: learning rate
    :param env: the environment
    :param gamma: discount factor
    :param num_episodes: number of training episodes
    :param epsilon_exploration: exploration rate
    :param epsilon_exploration_rule: rule for decaying exploration (if not None)
    :param epsilon_opt: degree of greediness of the optimal player
    :param test_freq: test frequency
    :param verbose: if True prints the statistics during training
    :return:
    """
    num_actions = 9

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    frame_count = 0

    model = create_q_model()
    model_target = create_q_model()

    # Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    turns = np.array(['X', 'O'])
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    loss_train = np.empty(num_episodes)
    if test_freq is not None:
        episode_Mopt = [measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=0.))]
        episode_Mrand = [measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=1.))]
    else:
        episode_Mopt = []  # initialize
        episode_Mrand = []  # initialize
    if verbose and (test_freq is not None):
        print('Episode  0 :\tM_opt = ', episode_Mopt[0], '\tM_rand = ', episode_Mrand[0])
    # Rule for exploration
    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    for itr in tqdm(range(num_episodes)):
        my_player = turns[itr % 2]
        player_opt = OptimalPlayer(epsilon=epsilon_opt, player=turns[(itr+1) % 2])
        state, _, _ = env.reset()
        if env.current_player == player_opt.player:
            move = player_opt.act(state)
            state, _, _ = env.step(move)
        # state = np.array(state)
        for i in range(num_actions):
            # frame_count += 1

            state_tensor = grid_to_tensor(state, my_player)
            state_tensor = tf.expand_dims(state_tensor, axis=0)
            # Use epsilon-greedy for exploration
            action = dqn_epsilon_greedy(model, state_tensor, epsilon_exploration_rule(itr+1))

            # Apply the sampled action in our environment
            try:
                state_adv, _, _ = env.step(action)
            except ValueError:
                env.end = True
                env.winner = player_opt.player
            if not env.end:
                action_adv = player_opt.act(state_adv)
                state_next, _, _ = env.step(action_adv)
            else:
                state_next = state  # better choice of this update

            reward = env.reward(player=my_player)
            done = env.end

            frame_count += 1
            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(grid_to_tensor(state, my_player))
            state_next_history.append(grid_to_tensor(state_next, my_player))
            done_history.append(done)
            rewards_history.append(reward)

            state = state_next
            # Update after every update_freq steps and once batch size is over 64
            if frame_count % update_freq == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = tf.stack([state_history[i] for i in indices], axis=0)
                state_next_sample = tf.stack([state_next_history[i] for i in indices], axis=0)
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = np.array([float(done_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target(state_next_sample, training=False).numpy()
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * np.amax(future_rewards, axis=1) * (1 - done_sample)

                # If final frame set the last value to -1
                # updated_q_values = updated_q_values * (1 - done_sample) #- done_sample

                # Create a mask as to calculate the loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]
            if done:
                break

        if itr % update_target_network == 0:
            # update the target network with new weights
            if verbose:
                print("******* Updating target network *******")
            model_target.set_weights(model.get_weights())

        episode_rewards[itr] = env.reward(player=my_player)
        if len(done_history) > batch_size:
            loss_train[itr] = loss
        # Testing the performance
        if (test_freq is not None) and ((itr+1) % test_freq == 0):
            M_opt = measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=0.))
            M_rand = measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=1.))
            episode_Mopt.append(M_opt)
            episode_Mrand.append(M_rand)
            if verbose:
                print('Episode ', itr+1, ':\tM_opt = ', M_opt, '\tM_rand = ', M_rand)

    # Dictionary of stats
    stats = {
        'rewards': episode_rewards,
        'test_Mopt': episode_Mopt,
        'test_Mrand': episode_Mrand,
        'loss': loss_train
    }
    return model, stats


def deep_q_learning_self_practice(env, lr=5e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                                  epsilon_exploration_rule=None, test_freq=None, verbose=False,
                                  batch_size=64, max_memory_length=10000, update_target_network=500, update_freq=1):
    """

    This function learns an agent to play Tic-Tac-Toe by training it with DQN against itself in a supervised fashion
    :param update_freq: update frequency (frames)
    :param update_target_network: how often to update the target network (num games)
    :param max_memory_length: replay buffer size
    :param batch_size: batch size for the sampling when updating
    :param lr: learning rate
    :param env: the environment
    :param gamma: discount factor
    :param num_episodes: number of training episodes
    :param epsilon_exploration: exploration rate
    :param epsilon_exploration_rule: rule for decaying exploration (if not None)
    :param test_freq: test frequency
    :param verbose: if True prints the statistics during training
    :return:
    """
    num_actions = 9

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    frame_count = 0

    model = create_q_model()
    model_target = create_q_model()

    # Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    turns = np.array(['X', 'O'])
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    loss_train = np.empty(num_episodes)
    if test_freq is not None:
        episode_Mopt = [measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=0.))]
        episode_Mrand = [measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=1.))]
    else:
        episode_Mopt = []  # initialize
        episode_Mrand = []  # initialize
    if verbose and (test_freq is not None):
        print('Episode  0 :\tM_opt = ', episode_Mopt[0], '\tM_rand = ', episode_Mrand[0])
    # Rule for exploration
    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    for itr in tqdm(range(num_episodes)):
        my_player = turns[itr % 2]
        curr_player = 'X'  # The one who makes the update
        env.reset()
        state, _, _ = env.observe()
        state_tensor = grid_to_tensor(state, 'X')   # X always makes the first move
        next_state = state  # Not to be used in the real updates
        # First step outside the loop
        action = dqn_epsilon_greedy(model, tf.expand_dims(state_tensor, axis=0), epsilon_exploration_rule(itr+1))
        state_adv, _, _ = env.step(action)
        state_adv_tensor = grid_to_tensor(state_adv, 'O')
        action_adv = dqn_epsilon_greedy(model, tf.expand_dims(state_adv_tensor, axis=0),
                                        epsilon_exploration_rule(itr+1))
        for i in range(num_actions):
            # Adversarial turn
            adv_player = 'X' if curr_player == 'O' else 'O'
            try:
                next_state, _, _ = env.step(action_adv)
                # next_state_tensor = grid_to_tensor(next_state, adv_player)
            except ValueError:
                env.end = True
                env.winner = curr_player
            done_adv = env.end
            if done_adv:  # The adversarial won the game or chose the wrong move
                reward_adv = env.reward(player=adv_player)
                action_history.append(action_adv)
                state_adv_tensor = grid_to_tensor(state_adv, adv_player)
                state_history.append(state_adv_tensor)
                state_next_history.append(state_adv_tensor)  # Will not be used
                done_history.append(done_adv)
                rewards_history.append(reward_adv)
            # curr_player's turn
            done = env.end
            reward = env.reward(player=curr_player)
            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(grid_to_tensor(state, curr_player))
            next_state_tensor = grid_to_tensor(next_state, curr_player)
            state_next_history.append(next_state_tensor)
            done_history.append(done)
            rewards_history.append(reward)

            # Preparing the next update (curr_player becomes the adversarial)
            curr_player = adv_player
            action = action_adv
            if not done:
                action_adv = dqn_epsilon_greedy(model, tf.expand_dims(next_state_tensor, axis=0),
                                                epsilon_exploration_rule(itr + 1))
            state = state_adv
            state_adv = next_state

            # Update after every update_freq steps and once batch size is over 64
            frame_count += 1
            if frame_count % update_freq == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = tf.stack([state_history[i] for i in indices], axis=0)
                state_next_sample = tf.stack([state_next_history[i] for i in indices], axis=0)
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = np.array([float(done_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target(state_next_sample, training=False).numpy()
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * np.amax(future_rewards, axis=1) * (1 - done_sample)

                # If final frame set the last value to -1
                # updated_q_values = updated_q_values * (1 - done_sample) #- done_sample

                # Create a mask as to calculate the loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]
                break
            assert (len(rewards_history) <= max_memory_length)

        if itr % update_target_network == 0:
            # update the target network with new weights
            if verbose:
                print("******* Updating target network *******")
            model_target.set_weights(model.get_weights())

        episode_rewards[itr] = env.reward(player=my_player)
        if len(done_history) > batch_size:
            loss_train[itr] = loss
        # Testing the performance
        if (test_freq is not None) and ((itr+1) % test_freq == 0):
            M_opt = measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=0.))
            M_rand = measure_performance(DeepQPlayer(model=model), OptimalPlayer(epsilon=1.))
            episode_Mopt.append(M_opt)
            episode_Mrand.append(M_rand)
            if verbose:
                print('Episode ', itr+1, ':\tM_opt = ', M_opt, '\tM_rand = ', M_rand)

    # Dictionary of stats
    stats = {
        'rewards': episode_rewards,
        'test_Mopt': episode_Mopt,
        'test_Mrand': episode_Mrand,
        'loss': loss_train
    }
    return model, stats


def deep_q_learning(env, lr=5e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                    epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False,
                    batch_size=64, max_memory_length=10000, update_target_network=500, update_freq=1,
                    against_opt=False, self_practice=False):
    """
    Calls deep_q_learning_against_opt if against_opt = True and deep_q_learning_self_practice if self_practice = True
    """
    if int(against_opt + self_practice) != 1:
        raise ValueError("Please, select a training method")
    if against_opt:
        return deep_q_learning_against_opt(env, lr, gamma, num_episodes, epsilon_exploration,
                                           epsilon_exploration_rule, epsilon_opt, test_freq, verbose,
                                           batch_size, max_memory_length, update_target_network, update_freq)
    if self_practice:
        return deep_q_learning_self_practice(env, lr, gamma, num_episodes, epsilon_exploration,
                                             epsilon_exploration_rule, test_freq, verbose,
                                             batch_size, max_memory_length, update_target_network, update_freq)


def deep_train_avg(var_name, var_values, deep_q_learning_params_list, num_avg=10, save_stats=True):
    """
    Function that computes all the quantities of interest averaging over many training runs
    :param var_name: name of the parameter
    :param var_values: values for the parameter var_name
    :param deep_q_learning_params_list: list of dictionaries with the parameters
        for the deep Q-learning for each value of var_name, for example {'test_freq': 250, 'self_practice': True, ...}
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
            deep_q_learning_params = deep_q_learning_params_list[idx]
            # perform Q-learning with the current parameters
            model, stats = deep_q_learning(**deep_q_learning_params)
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

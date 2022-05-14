from utils import *


def deep_q_learning_against_opt(env, lr=5e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                                epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False,
                                batch_size=64, max_memory_length=10000, update_target_network=500, update_freq=1):
    """
    This function learns an agent to play Tic-Tac-Toe by training it with DQN against the optimal strategy
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

    for itr in range(num_episodes):
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
            if epsilon_exploration_rule(itr+1) > np.random.uniform(0, 1):
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                action_probs = model(state_tensor, training=False)
                # Take best action
                max_indices = tf.where(action_probs[0] == tf.reduce_max(action_probs[0]))
                action = int(max_indices[np.random.randint(0, len(max_indices))])  # ties are split randomly

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
                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target(state_next_sample, training=False)  # WHAT CHANGES WITH PREDICT
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1) * (1 - done_sample)

                # If final frame set the last value to -1
                # updated_q_values = updated_q_values * (1 - done_sample) #- done_sample

                # Create a mask so we only calculate loss on the updated Q-values
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
        'loss_train': loss_train
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
    raise NotImplementedError


def deep_q_learning(env, lr=5e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                    epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False,
                    against_opt=False, self_practice=False):
    """

    """
    if int(against_opt + self_practice) != 1:
        raise ValueError("Please, select a training method")
    if against_opt:
        return deep_q_learning_against_opt(env, lr, gamma, num_episodes, epsilon_exploration,
                                           epsilon_exploration_rule, epsilon_opt, test_freq, verbose)
    if self_practice:
        return deep_q_learning_self_practice(env, lr, gamma, num_episodes, epsilon_exploration,
                                             epsilon_exploration_rule, test_freq, verbose)

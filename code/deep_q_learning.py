from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *


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
        - the tensor to be given as input to the network
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
        Performs a greedy move, i.e. a (1-epsilon)-greedy action with epsilon equal to zero
        :param self: self
        :param grid: current state
        :param kwargs: keyword arguments
        :return:
            - the action chosen greedily
        """
        grid = tf.expand_dims(grid_to_tensor(grid, self.player), axis=0)
        with tf.device("/device:cpu:0"):
            action_probs = self.model(grid, training=False)
        # Take best action
        max_indices = tf.where(action_probs[0] == tf.reduce_max(action_probs[0]))
        return int(max_indices[np.random.randint(0, len(max_indices))])  # ties are split randomly


def deep_q_learning_against_opt(env, lr=1e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                                epsilon_exploration_rule=None, epsilon_opt=0.5, test_freq=None, verbose=False,
                                batch_size=64, max_memory_length=10000, update_target_network=500, update_freq=1):
    """
    This function trains a Deep Q-agent to play Tic-Tac-Toe against the optimal strategy
    :param env: the environment
    :param lr: learning rate
    :param gamma: discount factor
    :param num_episodes: number of training episodes
    :param epsilon_exploration: exploration rate
    :param epsilon_exploration_rule: rule for decaying exploration (if not None)
    :param epsilon_opt: degree of greediness of the optimal player
    :param test_freq: test frequency
    :param verbose: if True prints the statistics during training
    :param update_freq: update frequency (frames)
    :param update_target_network: how often to update the target network (num games)
    :param max_memory_length: replay buffer size
    :param batch_size: batch size for the sampling when updating
    :return:
        - model: the model after training with DQN (i.e. the weights and biases of the MLP)
        - stats: dictionary of statistics collected during training
    """
    num_actions = 9

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    frame_count = 0

    model = create_q_model()  # create network
    model_target = create_q_model()  # create target network

    # Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # Using huber loss for stability (\delta = 1 by default)
    loss_function = keras.losses.Huber()

    turns = np.array(['X', 'O'])
    # Stats of training
    episode_rewards = np.empty(num_episodes)  # dummy initialization
    loss_train = 1e50 * np.ones(num_episodes)  # dummy initialization
    if test_freq is not None:
        # measure the performance against Opt(0) and Opt(1) for the initial model and weights initialization (baseline)
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

        for i in range(num_actions):

            state_tensor = grid_to_tensor(state, my_player)
            state_tensor = tf.expand_dims(state_tensor, axis=0)
            # Use epsilon-greedy for exploration
            action = dqn_epsilon_greedy(model, state_tensor, epsilon_exploration_rule(itr+1))

            # Apply the sampled action in our environment
            try:
                state_adv, _, _ = env.step(action)
            except ValueError:
                env.end = True  # the agent chose an unavailable action, thus reward = -1
                env.winner = player_opt.player
            if not env.end:
                action_adv = player_opt.act(state_adv)
                state_next, _, _ = env.step(action_adv)
            else:
                state_next = state  # not used anyway if the game has ended

            reward = env.reward(player=my_player)  # reward
            done = env.end

            frame_count += 1
            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(grid_to_tensor(state, my_player))
            state_next_history.append(grid_to_tensor(state_next, my_player))
            done_history.append(done)
            rewards_history.append(reward)

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            state = state_next
            # Update after every update_freq steps and there are enough samples in the replay buffer
            if frame_count % update_freq == 0 and len(done_history) >= batch_size:
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
                with tf.device("/device:cpu:0"):
                    future_rewards = model_target(state_next_sample, training=False)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1) * (1 - done_sample)  # targets

                # Create a mask as to calculate the loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    with tf.device("/device:cpu:0"):
                        # Train the model on the states and updated Q-values
                        q_values = model(state_sample)
                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if done:
                break

        if itr % update_target_network == 0:
            # update the target network with new weights
            if verbose:
                print("******* Updating target network *******")
            model_target.set_weights(model.get_weights())  # set new weights of the target net

        episode_rewards[itr] = env.reward(player=my_player)  # reward of the current episode
        if len(done_history) >= batch_size:
            loss_train[itr] = loss  # get training loss

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


def deep_q_learning_self_practice(env, lr=1e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
                                  epsilon_exploration_rule=None, test_freq=None, verbose=False,
                                  batch_size=64, max_memory_length=10000, update_target_network=500, update_freq=1):
    """

    This function learns an agent to play Tic-Tac-Toe by training it with DQN against itself in an unsupervised fashion
    :param env: the environment
    :param lr: learning rate
    :param gamma: discount factor
    :param num_episodes: number of training episodes
    :param epsilon_exploration: exploration rate
    :param epsilon_exploration_rule: rule for decaying exploration (if not None)
    :param test_freq: test frequency
    :param verbose: if True prints the statistics during training
    :param update_freq: update frequency (frames)
    :param update_target_network: how often to update the target network (num games)
    :param max_memory_length: replay buffer size
    :param batch_size: batch size for the sampling when updating
    :return:
        - model: the model after training with DQN (i.e. the weights and biases of the MLP)
        - stats: dictionary of statistics collected during training
    """
    num_actions = 9

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    frame_count = 0

    model = create_q_model()  # create network
    model_target = create_q_model()  # create target network

    # Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # Using huber loss for stability (\delta = 1 by default)
    loss_function = keras.losses.Huber()

    turns = np.array(['X', 'O'])
    # Stats of training
    episode_rewards = np.empty(num_episodes)  # dummy initialization
    loss_train = 1e50 * np.ones(num_episodes)  # dummy initialization
    if test_freq is not None:
        # measure the performance against Opt(0) and Opt(1) for the initial model and weights initialization (baseline)
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
                env.end = True  # the agent chose an unavailable action, thus reward = -1
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

            # Update after every update_freq steps and when enough samples are in the replay buffer
            frame_count += 1
            if frame_count % update_freq == 0 and len(done_history) >= batch_size:
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
                with tf.device("/device:cpu:0"):
                    future_rewards = model_target(state_next_sample, training=False)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1) * (1 - done_sample)  # targets

                # Create a mask as to calculate the loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    with tf.device("/device:cpu:0"):
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
            model_target.set_weights(model.get_weights())  # set new weights of the target net

        episode_rewards[itr] = env.reward(player=my_player)  # reward
        if len(done_history) >= batch_size:
            loss_train[itr] = loss  # training loss

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


def deep_q_learning(env, lr=1e-4, gamma=0.99, num_episodes=20000, epsilon_exploration=0.1,
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

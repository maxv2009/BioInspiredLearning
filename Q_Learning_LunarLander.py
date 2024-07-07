import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from collections import deque


# Create the model
def create_model(state_dim, action_dim):
    model = tf.keras.models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_dim)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    # Add experience to the buffer
    def add(self, experience):
        self.buffer.append(experience)

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        experience = map(np.array, zip(*batch))
        return experience

    # Get the size of the buffer
    def __len__(self):
        return len(self.buffer)


class DQN_Agent:
    def __init__(self, state_dim, action_dim, replay_buffer, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.q_network = create_model(state_dim, action_dim)
        self.target_network = create_model(state_dim, action_dim)
        self.target_network.set_weights(self.q_network.get_weights())
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

    # Select an action based on the epsilon-greedy policy
    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = np.reshape(state, [1, self.state_dim])
            q_values = self.q_network.predict(state)
            return np.argmax(q_values[0])

    # Train the agent by sampling experiences from the replay buffer
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        experience = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experience

        target_q = self.q_network.predict(states)
        next_q = self.target_network.predict(next_states)

        batch_index = np.arange(self.batch_size)

        # Update the Q-values using the Bellman equation
        target_q[batch_index, actions] = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)

        self.q_network.fit(states, target_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Update the target network
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

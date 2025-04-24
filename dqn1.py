import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import random

# 生成随机种子
seed = random.randint(0, 2**32 - 1)

# 保存种子到文件
seed_file_path = 'dqn_seed(323033直接分段2).txt'
with open(seed_file_path, 'w') as f:
    f.write(str(seed))

# 设置随机种子
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 经验回放类
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

# 奖励函数
class RewardFunction:
    def __init__(self, stability_factor=0.5):
        self.stability_factor = stability_factor
        self.recent_mses = []  # 存储最近三次的MSE值

    def calculate_mse(self, true_values, predicted_values):
        return np.mean((true_values - predicted_values) ** 2)

    def update_recent_mses(self, mse):
        self.recent_mses.append(mse)
        if len(self.recent_mses) > 3:
            self.recent_mses.pop(0)

    def calculate_stability_reward(self):
        if len(self.recent_mses) < 3:
            return 0
        else:
            return -np.mean(np.abs(np.diff(self.recent_mses)))

    def get_reward(self, true_values, predicted_values):
        mse = self.calculate_mse(true_values, predicted_values)
        self.update_recent_mses(mse)
        error_reward = -mse
        stability_reward = self.calculate_stability_reward()
        return (1 - self.stability_factor) * error_reward + self.stability_factor * stability_reward

# 定义DQN模型
class DQN:
    def __init__(self, state_size, action_size, reward_fn, max_weight_sum=2.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.99  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减因子
        self.epsilon_min = 0.01  # 最小探索率
        self.learning_rate = 0.001
        self.reward_fn = reward_fn
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.max_weight_sum = max_weight_sum

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, memory, batch_size):
        minibatch = memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        self.target_model.set_weights(self.model.get_weights())

def run_experiment():
    train_data = pd.read_csv('second_part(val).csv')
    # train_data = train_data.iloc[int(len(train_data) * 0.1):]  # 取后90%数据作为traindata
    train_data = np.array(train_data)

    train_cycles = train_data[:, 0]

    train_size = train_data.shape[0]
    state_size = 2
    action_size = 4
    epochs = 50
    batch_size = 32
    memory = ReplayBuffer(512)
    reward_fn = RewardFunction()
    dqn = DQN(state_size, action_size, reward_fn, max_weight_sum=2.0)

    actions = [[0.001, 0.001], [0.001, -0.001], [-0.001, 0.001], [-0.001, -0.001]]


    rewards = []
    best_reward = -float('inf')
    best_weights = None

    for e in range(epochs):
        total_reward = 0
        w1, w2 = 0.5, 0.5
        for i in range(train_size):
            state = np.array([[w1, w2]])
            action_index = dqn.act(state)
            action = actions[action_index]
            next_weights = np.add([w1, w2], action)
            next_weights = np.clip(next_weights, 0, dqn.max_weight_sum - np.sum(next_weights))
            next_state = np.array([next_weights])
            predicted_value = train_data[i][2] * next_weights[0] + train_data[i][3] * next_weights[1]
            reward = reward_fn.get_reward(train_data[i][1], predicted_value)
            total_reward += reward
            memory.add((state, action_index, reward, next_state, False))
            w1, w2 = next_weights

        dqn.replay(memory, batch_size)
        rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = [w1, w2]

        print(f"Epoch: {e+1}/{epochs}, Reward: {total_reward:.2f}, Epsilon: {dqn.epsilon:.2f}")

    return best_weights

# Run experiment and collect results
best_weights = run_experiment()

# Output the best weights
print(f"Best weights: {best_weights}")

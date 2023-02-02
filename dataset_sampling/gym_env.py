import gym
import numpy as np
import torch
import random
from collections import deque

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DQNnet(torch.nn.Module):
    def __init__(self, state_dim, feat_dim, action_dim,
                 gamma=0.95, epsilon_min=0.01, epsilon_decay=0.999, dtype=torch.float64):
        super(DQNnet, self).__init__()
        # nnet
        self.action_dim = action_dim
        self.linear1 = torch.nn.Linear(state_dim, feat_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(feat_dim, action_dim)
        self.dtype=dtype

        # dqn
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.95
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.linear1(torch.from_numpy(x).to(device).type(self.dtype))
        feature_vec = self.activation(x)
        out = self.linear2(feature_vec)
        return feature_vec, out

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            _, act_values = self.forward(state)
            return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                _, val = self.forward(next_state)
                target = (reward + self.gamma * torch.max(val).item())
            _, target_f = self.forward(state)
            target_c = target_f.clone().detach()
            target_c[action] = target
            loss = self.loss_fn(target_f, target_c)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
def build_dataset(args):
    num_episodes = args['length']
    dtype = args['dtype']
    gamma = args['gamma']
    feature_type = args['feature_type']

    if feature_type == 'kernel':
        result = get_kernel_features(args)
    elif feature_type == 'dqn':
        result = get_dqn_features(args)

    PHI = result['PHI']
    dataset = result['dataset']

    f_dim = PHI.shape[0]
    A_hat = torch.zeros((f_dim, f_dim), device=device, dtype=dtype)
    b_hat = torch.zeros((f_dim, 1), device=device, dtype=dtype)
    C_hat = torch.zeros((f_dim, f_dim), device=device, dtype=dtype)
    for s1, s2, r in dataset:
        A_hat = A_hat + PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
        b_hat = b_hat + r * PHI[:, [s1]]
        C_hat = C_hat + PHI[:, [s1]] @ PHI[:, [s1]].T
    A_hat = A_hat / num_episodes
    b_hat = b_hat / num_episodes
    C_hat = C_hat / num_episodes
    result['A_hat'] = A_hat
    result['C_hat'] = C_hat
    result['b_hat'] = b_hat

    A_C_hat_inv = A_hat @ torch.linalg.inv(C_hat)
    theta_est = torch.linalg.inv(A_C_hat_inv @ A_hat) @ A_C_hat_inv @ b_hat
    result['theta_est'] = theta_est

    return result


def get_kernel_features(args):
    env_name = args['gym_env']
    num_kernels = args['num_kernels']
    num_episodes = args['length']
    threshold = args['feat_threshold']
    dtype = args['dtype']

    env = gym.make(env_name)
    state = env.reset()
    data = []
    counter = 0
    while counter < num_episodes:
        action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)
        data.append( (state, new_state,  reward) )
        counter += 1
        state = new_state
        if done:
            new_state = env.reset()
            data.append( (state, new_state,  0) )
            counter += 1
            state = new_state
    data = data[:num_episodes]

    max_obs = np.max(np.vstack([np.vstack((x[0], x[1])) for x in data]), axis=0)
    min_obs = np.min(np.vstack([np.vstack((x[0], x[1])) for x in data]), axis=0)
    all_features = [x[0] for x in data]
    all_features.append(data[-1][1])
    all_features = np.vstack(all_features)

    means = []
    devs = []
    for i in range(1, num_kernels + 1):
        for j in range(all_features.shape[1]):
            mean_loc = np.quantile(all_features[:, j], i / (num_kernels + 1))
            dev_loc = np.max(np.abs([mean_loc - min_obs[j], max_obs[j] - mean_loc])) / 4
            means.append(mean_loc)
            devs.append(dev_loc)

    means = torch.from_numpy(np.array(means)).to(device)
    means = means.type(dtype)
    devs = torch.from_numpy(np.array(devs)).to(device)
    devs = devs.type(dtype)

    PHI = []
    dataset = []
    pointer = 0
    for (state1, state2, r) in data:
        dataset.append((pointer, pointer + 1, r))
        state = [state1 for _ in range(num_kernels)]
        state = np.hstack(state)
        state = torch.from_numpy(state).to(device)
        state = state.type(dtype)
        feat_vector = torch.exp(-((state - means) ** 2 / devs ** 2))
        PHI.append(feat_vector)
        pointer += 1

    (state1, state2, r) = data[-1]
    state = [state2 for _ in range(num_kernels)]
    state = np.hstack(state)
    state = torch.from_numpy(state).to(device)
    state = state.type(dtype)
    feat_vector = torch.exp(-((state - means) ** 2 / devs ** 2))
    PHI.append(feat_vector)

    PHI = torch.vstack(PHI)
    PHI = PHI.T
    PHI = torch.vstack([PHI, torch.ones(size=(1, num_episodes + 1), device=device)])
    PHI = PHI / (torch.norm(PHI, dim=0).max().item())

    while True:
        cor_matrix = np.abs(np.triu(np.corrcoef(PHI.cpu()), 1))
        if np.max(cor_matrix) >= threshold:
            ind = np.unravel_index(np.argmax(cor_matrix, axis=None), cor_matrix.shape)
            PHI = torch.vstack([PHI[:ind[1], ], PHI[ind[1] + 1:, :]])
        else:
            break

    return {'dataset': dataset, 'PHI': PHI}


def get_dqn_features(args):
    torch.set_default_dtype(args['dtype'])

    env_name = args['gym_env']
    feat_dim = args['feat_dim']
    num_episodes = args['length']
    threshold = args['feat_threshold']
    dtype = args['dtype']
    EPISODES = args['dqn_episodes']

    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNnet(state_size, feat_dim, action_size).to(device)
    batch_size = 32

    for e in range(1, EPISODES + 1):
        state = env.reset()
        for time in range(498):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -1
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == 498:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    print('Agent Training done')
    env = gym.make(env_name)
    state = env.reset()
    features, act_values = agent.forward(state)
    data = []
    counter = 0
    with torch.no_grad():
        while counter < num_episodes:
            action = torch.argmax(act_values).item()
            state, reward, done, _ = env.step(action)
            new_features, act_values = agent.forward(state)
            data.append((features, new_features, reward))
            counter += 1
            features = new_features
            if done:
                state = env.reset()
                new_features, act_values = agent.forward(state)
                data.append((features, new_features, 0))
                counter += 1
                features = new_features
        data = data[:num_episodes]

    PHI = [x[0] for x in data]
    PHI.append(data[-1][1])
    PHI = torch.vstack(PHI).T.detach()
    PHI = PHI - PHI.mean(dim=1, keepdims=True)
    dataset = []
    pointer = 0
    for (state1, state2, r) in data:
        dataset.append((pointer, pointer + 1, r))
        pointer += 1
    U, S, V = torch.pca_lowrank(PHI, center=True)
    sum_S = torch.sum(S)
    for keep_feat in range(1, S.shape[0]):
        if S[:keep_feat].sum() >= sum_S * threshold:
            break
    PHI = (PHI.T @ U[:, :keep_feat]).T
    PHI = torch.vstack([PHI, torch.ones(size=(1, num_episodes + 1), device=device)])
    PHI = PHI / (torch.norm(PHI, dim=0).max().item())
    return {'dataset': dataset, 'PHI': PHI}

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def build_dataset(args):
    env_name = args['gym_env']
    num_kernels = args['num_kernels']
    num_episodes = args['length']
    dtype = args['dtype']
    threshold = args['feat_threshold']
    gamma = args['gamma']

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

    result = {}
    result['dataset'] = dataset
    result['PHI'] = PHI

    result['dataset'] = dataset
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
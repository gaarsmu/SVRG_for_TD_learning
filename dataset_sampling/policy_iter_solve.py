import torch
import numpy as np
from utils import compute_f
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def solve(problem_data, args):
    theta_opt = problem_data['theta_est']
    dataset = problem_data['dataset']
    theta_init = problem_data['theta_init']
    gamma = args['gamma']
    PHI = problem_data['PHI']
    report_freq = args['report_freq']
    max_iter = args['max_iter']

    cur_dist = compute_f(problem_data, theta_init, theta_opt, args)
    dists = [cur_dist]
    norms = [torch.norm(theta_init - theta_opt).item()]

    if args['learning_rate'] == 'const':
        eta = 1
    elif args['learning_rate'] == 'root':
        eta = 1/int(np.sqrt(max_iter+1))
    elif args['learning_rate'] == 'decr':
        eta = 1
    elif args['learning_rate'] == 'decr_pow':
        eta = 1
    else:
        eta = args['learning_rate']

    theta_cur = theta_init
    for num_iter in range(1, max_iter+1):
        choice = np.random.choice(range(len(dataset)), size=1)
        s1, s2, r = dataset[choice[0]]
        update_loc = (r + gamma * PHI[:, [s2]].T @ theta_cur - PHI[:, [s1]].T @ theta_cur) * PHI[:, [s1]]

        if args['learning_rate'] == 'decr':
            eta = 1/num_iter
        elif args['learning_rate'] == 'decr_pow':
            eta = 1/(num_iter**args['decay_rate'])
        theta_cur = theta_cur + eta * update_loc

        if num_iter % report_freq == 0:
            dist = compute_f(problem_data, theta_cur, theta_opt, args)
            dists.append(dist)
            norms.append(torch.norm(theta_cur - theta_opt).item())

    result = {'distances': dists, 'norms': norms}
    return result

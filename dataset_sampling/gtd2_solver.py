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
        eta = args['lr_value']
    elif args['learning_rate'] == 'root':
        eta = 1/int(np.sqrt(max_iter+1))
    elif args['learning_rate'] == 'decr':
        eta = 1
    elif args['learning_rate'] == 'decr_pow':
        eta = 1
    else:
        eta = args['learning_rate']

    theta_cur = theta_init
    w_cur = torch.zeros_like(theta_cur)

    for num_iter in range(1, max_iter+1):
        choice = np.random.choice(range(len(dataset)), size=1)
        s1, s2, r = dataset[choice[0]]
        A_loc = PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
        b_loc = r * PHI[:, [s1]]
        C_loc = PHI[:, [s1]] @ PHI[:, [s1]].T

        update_loc_theta = -A_loc.T @ w_cur
        update_loc_w = A_loc @ theta_cur - b_loc + C_loc @ w_cur

        if args['learning_rate'] == 'decr':
            eta = 1/num_iter
        elif args['learning_rate'] == 'decr_pow':
            eta = 1/(num_iter**args['decay_rate'])
        beta = eta

        theta_cur = theta_cur - eta * update_loc_theta
        w_cur = w_cur - beta * update_loc_w

        if num_iter % report_freq == 0:
            dist = compute_f(problem_data, theta_cur, theta_opt, args)
            dists.append(dist)
            norms.append(torch.norm(theta_cur - theta_opt).item())

    result = {'distances': dists, 'norms': norms}
    return result
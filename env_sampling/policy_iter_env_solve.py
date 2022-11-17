import torch
import numpy as np
from utils import compute_f_env
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def solve(env, args):
    theta_opt = env.theta_opt
    theta_init = args['theta_init']
    gamma = args['gamma']
    PHI = env.PHI
    max_iter = args['max_iter']
    report_freq = args['report_freq']
    milestones = [0]

    cur_dist = compute_f_env(env, theta_init, theta_opt)
    dists = [cur_dist]
    norms = [torch.norm(theta_init - theta_opt).item()]

    if args['learning_rate'] == 'const':
        eta = args['lr_value']
    elif args['learning_rate'] == 'root':
        eta = 1/int(np.sqrt(milestones[-1]+1))
    elif args['learning_rate'] == 'decr':
        eta = 1
    elif args['learning_rate'] == 'decr_pow':
        eta = 1
    else:
        eta = args['learning_rate']

    theta_cur = theta_init
    for num_iter in range(1, max_iter+1):
        s1, s2, r = env.sample(1)[0]
        update_loc = (r + gamma * PHI[:, [s2]].T @ theta_cur - PHI[:, [s1]].T @ theta_cur) * PHI[:, [s1]]

        if args['learning_rate'] == 'decr':
            eta = 1/num_iter
        elif args['learning_rate'] == 'decr_pow':
            eta = 1/(num_iter**args['decay_rate'])
        theta_cur = theta_cur + eta * update_loc

        if num_iter % report_freq == 0:
            dist = compute_f_env(env, theta_cur, theta_opt)
            dists.append(dist)
            norms.append(torch.norm(theta_cur - theta_opt).item())
            milestones.append(num_iter)

    result = {'distances': dists, 'norms': norms}
    return result
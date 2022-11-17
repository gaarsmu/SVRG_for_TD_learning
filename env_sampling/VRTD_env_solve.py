import torch
from utils import compute_f_env
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def solve(env, args):
    theta_opt = env.theta_opt
    theta_init = args['theta_init']
    gamma = args['gamma']
    PHI = env.PHI
    M = args['batch_size']
    report_freq = args['report_freq']
    max_iter = args['max_iter']
    alpha = args['learning_rate']
    R = env.R

    cur_dist = compute_f_env(env, theta_init, theta_opt)
    distances = [cur_dist]
    norms = [torch.norm(theta_init - theta_opt).item()]
    milestones = [0]
    theta_cur = theta_init
    glob_counter = 0

    for update_num in range(max_iter//report_freq):
        theta_epoch = torch.clone(theta_cur)
        batch = env.sample(M)
        update_full = ((R[[x[0] for x in batch], [x[1] for x in batch]].reshape(M, 1) +
                        gamma * PHI[:, [x[1] for x in batch]].T @ theta_epoch -
                        PHI[:, [x[0] for x in batch]].T @ theta_epoch).T *
                       PHI[:, [x[0] for x in batch]]).mean(dim=1, keepdims=True)
        for s1, s2, r in batch:
            update_epoch = (r + gamma*PHI[:, [s2]].T@theta_epoch - PHI[:, [s1]].T@theta_epoch) * PHI[:, [s1]]
            update_loc = (r + gamma*PHI[:, [s2]].T@theta_cur - PHI[:, [s1]].T@theta_cur) * PHI[:, [s1]]
            theta_cur = theta_cur + alpha*(update_loc - update_epoch + update_full)

        glob_counter += 2*M
        dist = compute_f_env(env, theta_cur, theta_opt)
        distances.append(dist)
        norms.append(torch.norm(theta_cur - theta_opt).item())
        milestones.append(glob_counter)

    result = {'distances': distances, 'milestones': milestones, 'norms': norms}
    return result

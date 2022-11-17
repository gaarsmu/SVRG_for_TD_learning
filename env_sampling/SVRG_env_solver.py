import torch
import numpy as np
from utils import compute_f_env
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def solve(env, args):
    theta_init = args['theta_init']
    theta_opt = env.theta_opt
    gamma = args['gamma']
    conv_rate = args['conv_rate']
    report_freq = args['report_freq']
    PHI = env.PHI

    eig_min = torch.linalg.eigvals(1/2*(env.A+env.A.T)).real.min().item()
    eta = 1/8
    m = 16/eig_min
    #print('Number of updates in epoch:', int(m)+1)

    cur_dist = compute_f_env(env, theta_init, theta_opt)
    distances = [cur_dist]
    thetas = [theta_init]
    norms = [torch.norm(theta_init - theta_opt).item()]
    milestones = [0]
    epoch_nums = []
    theta_cur = theta_init
    glob_counter = 0

    R = env.R
    R_max = torch.max(torch.abs(R))

    for num_outer in range(1, args['num_outer']+1):
        theta_epoch = torch.clone(theta_cur)

        S2 = 4*R_max**2 + (2+4*gamma**2) * torch.norm(theta_epoch)**2
        batch_size = int((S2/conv_rate**(2*num_outer-2)).item())+1

        batch = env.sample(batch_size)
        update_full = ((R[[x[0] for x in batch], [x[1] for x in batch]].reshape(batch_size, 1) +
                        gamma * PHI[:, [x[1] for x in batch]].T @ theta_epoch -
                        PHI[:, [x[0] for x in batch]].T @ theta_epoch).T *
                       PHI[:, [x[0] for x in batch]]).mean(dim=1, keepdims=True)

        if glob_counter // report_freq != (glob_counter+batch_size) // report_freq:
            dist = compute_f_env(env, theta_cur, theta_opt)
            freq_counter = 0
            while (glob_counter+freq_counter*report_freq) // report_freq != (glob_counter+batch_size) // report_freq:
                distances.append(dist)
                thetas.append(theta_cur)
                norms.append(torch.norm(theta_cur - theta_opt).item())
                milestones.append((((glob_counter+freq_counter*report_freq) // report_freq + 1) * report_freq))
                freq_counter += 1

        glob_counter += batch_size

        num_updates_loc = np.random.randint(1, int(m)+2)
        for num_inner in range(num_updates_loc):
            s1, s2, r = env.sample(1)[0]
            update_epoch = (r + gamma*PHI[:, [s2]].T@theta_epoch - PHI[:, [s1]].T@theta_epoch) * PHI[:, [s1]]
            update_loc = (r + gamma*PHI[:, [s2]].T@theta_cur - PHI[:, [s1]].T@theta_cur) * PHI[:, [s1]]

            theta_cur = theta_cur + eta*(update_loc - update_epoch + update_full)
            glob_counter += 1
            if glob_counter % report_freq == 0:
                dist = compute_f_env(env, theta_cur, theta_opt)
                distances.append(dist)
                thetas.append(theta_cur)
                norms.append(torch.norm(theta_cur - theta_opt).item())
                milestones.append(glob_counter)
        epoch_nums.append(glob_counter)

    result = {'distances': distances, 'thetas': thetas, 'milestones': milestones, 'norms': norms, 'epochs' : epoch_nums}
    return result



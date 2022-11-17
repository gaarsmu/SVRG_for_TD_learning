import torch
import numpy as np
from utils import compute_f
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def solve(problem_data, args):
    dtype = args['dtype']
    theta_init = problem_data['theta_init']
    theta_opt = problem_data['theta_est']
    dataset = problem_data['dataset']
    gamma = args['gamma']
    PHI = problem_data['PHI']
    conv_rate = args['conv_rate']
    report_freq = args['report_freq']
    max_iter = args['max_iter']

    eig_min = torch.linalg.eigvals(1/2*(problem_data['A_hat']+problem_data['A_hat'].T)).real.min().item()
    eta = 1/8
    m = 16/eig_min
    print('Number of updates in epoch:', int(m)+1)

    cur_dist = compute_f(problem_data, theta_init, theta_opt, args)
    distances = [cur_dist]
    thetas = [theta_init]
    norms = [torch.norm(theta_init - theta_opt).item()]
    milestones = [0]
    theta_cur = theta_init
    glob_counter = 0

    R = torch.tensor([x[2] for x in dataset], device=device, dtype=dtype).reshape((len(dataset), 1))
    R_max = torch.max(R)

    for num_outer in range(1, args['num_outer']+1):
        theta_epoch = torch.clone(theta_cur)

        S2 = 4*R_max**2 + (2+4*gamma**2) * torch.norm(theta_epoch)**2
        batch_size = int((S2/conv_rate**(2*num_outer-2)).item())+1
        if batch_size >= len(dataset):
            update_full = ((R + gamma * PHI[:, [x[1] for x in dataset]].T @ theta_epoch -
                                         PHI[:, [x[0] for x in dataset]].T @ theta_epoch).T *
                                       PHI[:, [x[0] for x in dataset]]).mean(dim=1, keepdims=True)
            batch_size = len(dataset)
        else:
            batch_ind = np.random.choice(len(dataset), size=batch_size, replace=False)
            batch = [dataset[x] for x in batch_ind]
            update_full = ((R[batch_ind, :] + gamma * PHI[:, [x[1] for x in batch]].T @ theta_epoch -
                            PHI[:, [x[0] for x in batch]].T @ theta_epoch).T *
                           PHI[:, [x[0] for x in batch]]).mean(dim=1, keepdims=True)

        if glob_counter // report_freq != (glob_counter+batch_size) // report_freq:
            dist = compute_f(problem_data, theta_cur, theta_opt, args)
            freq_counter = 0
            while (glob_counter+freq_counter*report_freq) // report_freq != (glob_counter+batch_size) // report_freq:
                distances.append(dist)
                thetas.append(theta_cur)
                norms.append(torch.norm(theta_cur - theta_opt).item())
                milestones.append((((glob_counter+freq_counter*report_freq) // report_freq + 1) * report_freq))
                freq_counter += 1

        glob_counter += batch_size

        if glob_counter > max_iter:
            break

        num_updates_loc = np.random.randint(1, int(m)+2)
        for num_inner in range(num_updates_loc):
            choice = np.random.choice(range(len(dataset)), size=1)
            s1, s2, r = dataset[choice[0]]
            update_epoch = (r + gamma*PHI[:, [s2]].T@theta_epoch - PHI[:, [s1]].T@theta_epoch) * PHI[:, [s1]]
            update_loc = (r + gamma*PHI[:, [s2]].T@theta_cur - PHI[:, [s1]].T@theta_cur) * PHI[:, [s1]]

            theta_cur = theta_cur + eta*(update_loc - update_epoch + update_full)
            glob_counter += 1
            if glob_counter % report_freq == 0:
                dist = compute_f(problem_data, theta_cur, theta_opt, args)
                distances.append(dist)
                thetas.append(theta_cur)
                milestones.append(glob_counter)
                norms.append(torch.norm(theta_cur - theta_opt).item())
            if glob_counter > max_iter:
                break
    result = {'distances': distances, 'thetas': thetas, 'milestones': milestones, 'norms': norms}
    return result
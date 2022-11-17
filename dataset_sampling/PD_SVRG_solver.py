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
    report_freq = args['report_freq']
    max_iter = args['max_iter']

    w_init = torch.zeros_like(theta_init)

    A_hat = problem_data['A_hat']
    b_hat = problem_data['b_hat']
    C_hat = problem_data['C_hat']
    eig_min = torch.linalg.eigvals(1/2*(A_hat+A_hat.T)).real.min().item()
    ACA = A_hat.T @ torch.linalg.inv(C_hat) @ A_hat
    eig_ACA = torch.linalg.eigvals(ACA)
    eig_max_ACA = eig_ACA.real.max().item()
    eig_min_ACA = eig_ACA.real.min().item()

    eig_C = torch.linalg.eigvals(C_hat)
    eig_max_C = eig_C.real.max().item()
    eig_min_C = eig_C.real.min().item()
    beta = 8 * eig_max_ACA / eig_min_C

    G_hat = torch.zeros((A_hat.shape[0] * 2, A_hat.shape[0] * 2), dtype=dtype, device=device)

    for s1, s2, r in dataset:
        A_loc = PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
        C_loc = PHI[:, [s1]] @ PHI[:, [s1]].T
        G_loc = torch.cat([torch.cat([torch.zeros_like(A_loc), -np.sqrt(beta) * A_loc.T], dim=1),
                           torch.cat([np.sqrt(beta) * A_loc.T, beta * C_loc], dim=1)])
        G_hat = G_hat + G_loc.T @ G_loc
    G_hat = G_hat / len(dataset)
    L_G = torch.norm(G_hat)
    theor_learn_rate = eig_min_ACA/(48*(eig_max_C/eig_min_C)*L_G**2)
    theor_m = 51*(eig_max_C/eig_min_C)**2*L_G**2/eig_min_ACA**2

    if args['learning_rate'] == 'const':
        eta_w = args['lr_value_w']
        eta = args['lr_value_theta']
    elif args['learning_rate'] == 'mult_ratio':
        eta_w = args['lr_value_w']/eig_max_C
        eta = eta_w * args['lr_ratio']
    elif args['learning_rate'] == 'mult_mult':
        eta_w = args['lr_value_w'] / eig_max_C
        eta = args['lr_value_theta']*eig_min_C/(eig_max_C*eig_max_ACA)

    m = 16/eig_min

    cur_dist = compute_f(problem_data, theta_init, theta_opt, args)
    distances = [cur_dist]
    thetas = [theta_init]
    milestones = [0]
    norms = [torch.norm(theta_init - theta_opt).item()]
    theta_cur = theta_init
    w_cur = w_init
    glob_counter = 0
    num_outer = 0
    while True:
        num_outer += 1
        theta_epoch = torch.clone(theta_cur)
        w_epoch = torch.clone(w_cur)

        update_full_theta = -A_hat.T @ w_epoch
        update_full_w = A_hat@theta_epoch - b_hat + C_hat@w_epoch

        if glob_counter // report_freq != (glob_counter+len(dataset)) // report_freq:
            dist = compute_f(problem_data, theta_cur, theta_opt, args)
            freq_counter = 0
            while (glob_counter+freq_counter*report_freq) // report_freq != (glob_counter+len(dataset)) // report_freq:
                distances.append(dist)
                thetas.append(theta_cur)
                norms.append(torch.norm(theta_cur - theta_opt).item())
                milestones.append((((glob_counter+freq_counter*report_freq) // report_freq + 1) * report_freq))
                freq_counter += 1

        glob_counter += len(dataset)
        if glob_counter > max_iter:
            break
        num_updates_loc = np.random.randint(1, int(m)+2)
        for num_inner in range(num_updates_loc):
            choice = np.random.choice(range(len(dataset)), size=1)
            s1, s2, r = dataset[choice[0]]
            A_loc = PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
            b_loc = r * PHI[:, [s1]]
            C_loc = PHI[:, [s1]] @ PHI[:, [s1]].T

            update_epoch_theta = -A_loc.T @ w_epoch
            update_loc_theta = -A_loc.T @ w_cur

            update_epoch_w = A_loc@theta_epoch - b_loc + C_loc@w_epoch
            update_loc_w = A_loc @ theta_cur - b_loc + C_loc @ w_cur

            theta_cur = theta_cur - eta*(update_loc_theta - update_epoch_theta + update_full_theta)
            w_cur = w_cur - eta_w*(update_loc_w-update_epoch_w+update_full_w)

            glob_counter += 1
            if glob_counter % report_freq == 0:
                dist = compute_f(problem_data, theta_cur, theta_opt, args)
                distances.append(dist)
                thetas.append(theta_cur)
                milestones.append(glob_counter)
                norms.append(torch.norm(theta_cur - theta_opt).item())
            if glob_counter > max_iter:
                break
        if glob_counter > max_iter:
            break
    result = {'distances': distances, 'thetas': thetas, 'milestones': milestones,
              'theor_lr' : theor_learn_rate, 'theor_m' : theor_m, 'norms': norms}
    return result
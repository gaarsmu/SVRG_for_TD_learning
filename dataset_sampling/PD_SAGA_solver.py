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
    f_dim = args['f_dim']
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

    G = []
    for s1, s2, r in dataset:
        A_loc = PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
        b_loc = PHI[:, [s1]]*r
        C_loc = PHI[:, [s1]] @ PHI[:, [s1]].T

        theta_loc = -A_loc.t @ w_init
        w_loc = A_loc@theta_init - b_loc + C_loc@w_init

        upd_loc = torch.cat([theta_loc, w_loc], dim=0)
        G.append(upd_loc)
    G = torch.cat(G, dim=1)
    B = torch.mean(G, dim=1, keepdim=True)

    if args['learning_rate'] == 'const':
        eta_w = args['lr_value_w']
        eta = args['lr_value_theta']
    elif args['learning_rate'] == 'mult_ratio':
        eta_w = args['lr_value_w']/eig_max_C
        eta = eta_w * args['lr_ratio']
    elif args['learning_rate'] == 'mult_mult':
        eta_w = args['lr_value_w'] / eig_max_C
        eta = args['lr_value_theta']*eig_min_C/(eig_max_C*eig_max_ACA)

    if args['batch_size'] == '16/eig_min':
        m = 16/eig_min
    else:
        m = len(dataset)*args['batch_size']

    cur_dist = compute_f(problem_data, theta_init, theta_opt, args)
    distances = [cur_dist]
    thetas = [theta_init]
    milestones = [0]
    norms = [torch.norm(theta_init - theta_opt).item()]

    theta_cur = theta_init
    w_cur = w_init

    for num_iter in range(1, max_iter+1):
        choice = np.random.choice(range(len(dataset)), size=1)
        s1, s2, r = dataset[choice[0]]

        corr_loc = B - G[:, choice]

        A_loc = PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
        b_loc = r * PHI[:, [s1]]
        C_loc = PHI[:, [s1]] @ PHI[:, [s1]].T

        upd_theta_loc = -A_loc.t @ w_cur
        upd_w_loc = A_loc@theta_cur - b_loc + C_loc@w_cur

        theta_cur = eta*(corr_loc[:, :f_dim] + upd_theta_loc)
        w_cur = eta_w*(corr_loc[:, f_dim:] + upd_w_loc)

        B = B + (torch.stack([upd_theta_loc, upd_theta_loc],dim=0) - G[:, choice])
        G[:, choice] = torch.stack([upd_theta_loc, upd_theta_loc],dim=0)

        if num_iter % report_freq == 0:
            dist = compute_f(problem_data, theta_cur, theta_opt, args)
            distances.append(dist)
            norms.append(torch.norm(theta_cur - theta_opt).item())

    result = {'distances': distances, 'thetas': thetas, 'milestones': milestones,
              'norms': norms}
    return result
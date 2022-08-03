import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def build_dataset(args):
    dim = args['dimentionality']
    P = torch.normal(0, 1, size=(dim, dim)).to(device)
    P = (torch.exp(P)/torch.exp(P).sum(dim=1, keepdim=True)).T
    e, v = torch.linalg.eig(P)
    stat_dist = v[:, [0]].real
    if stat_dist.sum() < 0:
        stat_dist = stat_dist*-1.
    stat_dist = stat_dist/stat_dist.sum()
    stat_dist = stat_dist.reshape(-1, )
    assert torch.dist(stat_dist, P @ stat_dist) < 1e-4
    P = P.T
    result = {'P' : P, 'stat_dist' : stat_dist.reshape(-1, )}

    R = torch.rand(size=(dim, dim)).to(device)
    R = R/R.sum(dim=1, keepdim=True)
    result['R'] = R

    states = torch.arange(dim)
    dataset = []
    cur_state = states[torch.multinomial(stat_dist, 1).item()]
    for _ in range(args['num_episodes']):
        next_state = states[torch.multinomial(P[[cur_state], :], 1).item()]
        dataset.append( (cur_state.item(), next_state.item(), R[cur_state, next_state].item() ) )
        cur_state = next_state
    result['dataset'] = dataset

    if args['solve']:
        gamma = args['gamma']
        R_pi = (R * P).sum(dim=1, keepdim=True)
        V_pi = torch.linalg.inv(torch.eye(dim, device=device) - gamma * P) @ R_pi
        result['V_pi_true'] = V_pi

        PHI = torch.eye(dim, device=device)
        A_hat = torch.zeros((dim, dim), device=device)
        b_hat = torch.zeros((dim,1), device=device)
        C_hat = torch.zeros((dim, dim), device=device)
        for s1, s2, r in dataset:
            A_hat += PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
            b_hat += r * PHI[:, [s1]]
            C_hat += PHI[:, [s1]] @ PHI[:, [s1]].T
        A_hat = A_hat/args['num_episodes']
        b_hat = b_hat/args['num_episodes']
        C_hat = C_hat/args['num_episodes']

        A_C_hat_inv = A_hat @ torch.linalg.inv(C_hat)
        V_est = torch.linalg.inv(A_C_hat_inv @ A_hat) @ A_C_hat_inv @ b_hat
        result['V_est'] = V_est

    return result

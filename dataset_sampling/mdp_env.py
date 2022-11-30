import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def build_dataset(args):
    dim = args['dim']
    f_dim = args['f_dim']
    dtype = args['dtype']
    probs_type = args.get('probs_type', 'uniform')
    num_episodes = args["length"]
    num_actions = args.get('num_actions', 10)

    if probs_type == 'uniform':
        P = torch.normal(0, 1, size=(dim, dim), dtype=dtype).to(device)
        P = (torch.exp(P)/torch.exp(P).sum(dim=1, keepdim=True)).T
    elif probs_type == 'sparse':
        P = torch.rand(size=(dim, dim), dtype=dtype).to(device)
        rand_mat = torch.rand(dim, dim, dtype=dtype).to(device)
        k_th_quant = torch.topk(rand_mat, num_actions, largest=False)[0][:, -1:]
        mask = rand_mat <= k_th_quant
        P = P*mask + 10e-5
        P = (P/P.sum(dim=1, keepdim=True)).T
    elif probs_type == 'rooms':
        P = torch.zeros((dim, dim), dtype=dtype).to(device)
        cur_ind = 0
        finished = False
        sizes = []
        while not finished:
            size = torch.randint(low=5, high=15, size=(1,)).item()
            if cur_ind + size >= dim:
                size = dim - cur_ind
                finished = True
            sizes.append(size)
            P[cur_ind:cur_ind + size, cur_ind:cur_ind + size] = torch.rand(size=(size, size), dtype=dtype).to(device)
            cur_ind = cur_ind + size
        P = (P / P.sum(dim=1, keepdim=True))

        cur_ind = 0
        for size in sizes[:-1]:
            P[cur_ind + size - 1, cur_ind + size] = 1.
            P[cur_ind + size, cur_ind + size - 1] = 1.
            P[cur_ind + size - 1, :] = P[cur_ind + size - 1, :] / P[cur_ind + size - 1, :].sum()
            P[cur_ind + size, :] = P[cur_ind + size, :] / P[cur_ind + size, :].sum()
            cur_ind = cur_ind + size
        P = P.T


    e, v = torch.linalg.eig(P)
    stat_ind = e.real.argmax()
    stat_dist = v[:, [stat_ind]].real
    if stat_dist.sum() < 0:
        stat_dist = stat_dist*-1.
    stat_dist = stat_dist/stat_dist.sum()
    stat_dist = stat_dist.reshape(-1, )
    assert torch.dist(stat_dist, P @ stat_dist) < 1e-4
    P = P.T
    result = {'P': P, 'stat_dist': stat_dist.reshape(-1, )}

    if probs_type == 'rooms':
        cur_ind = 0
        room_mults = []
        R = torch.zeros(size=(dim, dim), dtype=dtype).to(device)
        for size in sizes:
            room_mult = torch.randint(low=1, high=11, size=(1,)).item()
            room_mults.append(room_mult)
            R[cur_ind:cur_ind + size, cur_ind:cur_ind + size] = torch.rand(size=(size, size), dtype=dtype).to(device)*room_mult
            cur_ind += size
            result['room_multipliers'] = room_mults
            result['room_sizes'] = sizes
    else:
        R = torch.rand(size=(dim, dim), dtype=dtype).to(device)
    result['R'] = R

    states = torch.arange(dim)
    dataset = []
    cur_state = states[torch.multinomial(stat_dist, 1).item()]
    for _ in range(num_episodes):
        next_state = states[torch.multinomial(P[[cur_state], :], 1).item()]
        dataset.append((cur_state.item(), next_state.item(), R[cur_state, next_state].item()))
        cur_state = next_state
    result['dataset'] = dataset

    if args['PHI'] == 'tabular':
        PHI = torch.eye(f_dim, device=device, dtype=dtype)
    elif args['PHI'] == 'random':
        PHI = torch.rand(size=(f_dim-1, dim), device=device, dtype=dtype)
        PHI = torch.vstack([PHI, torch.ones(size=(1,dim), device=device)])
        PHI = PHI/(torch.norm(PHI, dim=0).max().item())
    # if probs_type == 'rooms':
    #     PHI_rooms = torch.zeros(size=(len(sizes), dim), device=device, dtype=dtype)
    #     cur_ind = 0
    #     pointer = 0
    #     for size in sizes:
    #         PHI_rooms[pointer, cur_ind:cur_ind+size] = 1.
    #         pointer += 1
    #         cur_ind += size
    #     PHI_rand = torch.rand(size=(f_dim-1-len(sizes), dim), device=device, dtype=dtype)
    #     PHI = torch.vstack([PHI_rooms, PHI_rand,torch.ones(size=(1, dim), device=device)])
    #     print(len(sizes), (torch.norm(PHI, dim=0).max().item()))
    #     PHI = PHI / (torch.norm(PHI, dim=0).max().item())
    result['PHI'] = PHI

    gamma = args['gamma']
    R_pi = (R * P).sum(dim=1, keepdim=True)
    theta_pi = torch.linalg.inv(torch.eye(dim, device=device) - gamma * P) @ R_pi
    result['theta_pi_true'] = theta_pi

    A_hat = torch.zeros((f_dim, f_dim), device=device, dtype=dtype)
    b_hat = torch.zeros((f_dim, 1), device=device, dtype=dtype)
    C_hat = torch.zeros((f_dim, f_dim), device=device, dtype=dtype)
    for s1, s2, r in dataset:
        A_hat = A_hat + PHI[:, [s1]] @ (PHI[:, [s1]] - gamma * PHI[:, [s2]]).T
        b_hat = b_hat + r * PHI[:, [s1]]
        C_hat = C_hat + PHI[:, [s1]] @ PHI[:, [s1]].T
    A_hat = A_hat/num_episodes
    b_hat = b_hat/num_episodes
    C_hat = C_hat/num_episodes
    result['A_hat'] = A_hat
    result['C_hat'] = C_hat
    result['b_hat'] = b_hat

    A_C_hat_inv = A_hat @ torch.linalg.inv(C_hat)
    theta_est = torch.linalg.inv(A_C_hat_inv @ A_hat) @ A_C_hat_inv @ b_hat
    result['theta_est'] = theta_est

    return result

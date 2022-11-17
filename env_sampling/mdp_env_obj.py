import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class mdp_env():

    def __init__(self, args):
        dim = args['dim']
        f_dim = args['f_dim']
        dtype = args['dtype']
        probs_type = args['probs_type']
        self.gamma = args['gamma']

        if probs_type == 'uniform':
            P = torch.normal(0, 1, size=(dim, dim), dtype=dtype).to(device)
            P = (torch.exp(P) / torch.exp(P).sum(dim=1, keepdim=True)).T
        elif probs_type == 'sparse':
            P = torch.rand(size=(dim, dim), dtype=dtype).to(device)
            rand_mat = torch.rand(dim, dim, dtype=dtype).to(device)
            k_th_quant = torch.topk(rand_mat, 10, largest=False)[0][:, -1:]
            mask = rand_mat <= k_th_quant
            P = P * mask + 1e-5
            P = (P / P.sum(dim=1, keepdim=True)).T
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
                P[cur_ind:cur_ind + size, cur_ind:cur_ind + size] = torch.rand(size=(size, size), dtype=dtype).to(
                    device)
                cur_ind = cur_ind + size
            P = (P / P.sum(dim=1, keepdim=True))

            cur_ind = 0
            for size in sizes[:-1]:
                P[cur_ind + size - 1, cur_ind + size] = 1.
                P[cur_ind + size, cur_ind + size - 1] = 1.
                P[cur_ind + size - 1, :] = P[cur_ind + size - 1, :] / P[cur_ind + size - 1, :].sum()
                P[cur_ind + size, :] = P[cur_ind + size, :] / P[cur_ind + size, :].sum()
                cur_ind = cur_ind + size
            P[0,-1] = 1.
            P[-1,0] = 1.
            P[0,:] = P[0,:]/P[0,:].sum()
            P[-1, :] = P[-1, :] / P[-1, :].sum()
            P = P.T

        e, v = torch.linalg.eig(P)
        stat_ind = e.real.argmax()
        stat_dist = v[:, [stat_ind]].real
        if stat_dist.sum() < 0:
            stat_dist = stat_dist * -1.
        stat_dist = stat_dist / stat_dist.sum()
        stat_dist = stat_dist.reshape(-1, )
        assert torch.dist(stat_dist, P @ stat_dist) < 1e-4
        self.P = P.T
        self.stat_dist = stat_dist
        self.D = stat_dist * torch.eye(dim, dtype=dtype, device=device)
        self.states = torch.arange(dim)

        if probs_type == 'rooms':
            cur_ind = 0
            room_mults = []
            R = torch.zeros(size=(dim, dim), dtype=dtype).to(device)
            for size in sizes:
                room_mult = torch.randint(low=1, high=11, size=(1,)).item()
                room_mults.append(room_mult)
                R[cur_ind:cur_ind + size, cur_ind:cur_ind + size] = torch.rand(size=(size, size), dtype=dtype).to(
                    device) * room_mult
                cur_ind += size
                self.rooms_mult = room_mults
                self.room_sizes = sizes
        else:
            R = torch.rand(size=(dim, dim), dtype=dtype).to(device)
        self.R = R


        if probs_type != 'rooms':
            if args['PHI'] == 'tabular':
                PHI = torch.eye(f_dim, device=device, dtype=dtype)
            elif args['PHI'] == 'random':
                PHI = torch.rand(size=(f_dim - 1, dim), device=device, dtype=dtype)
                PHI = torch.vstack([PHI, torch.ones(size=(1, dim), device=device)])
                PHI = PHI / (torch.norm(PHI, dim=0).max().item())
        else:
            PHI_rooms = 0.5*torch.ones(size=(len(sizes), dim), device=device, dtype=dtype)
            cur_ind = 0
            pointer = 0
            for size in sizes:
                PHI_rooms[pointer, cur_ind:cur_ind+size] = 1.
                pointer += 1
                cur_ind += size
            PHI_rand = torch.rand(size=(f_dim-1-len(sizes), dim), device=device, dtype=dtype)
            PHI = torch.vstack([PHI_rooms, PHI_rand,torch.ones(size=(1, dim), device=device)])
            PHI = PHI / (torch.norm(PHI, dim=0).max().item())
        self.PHI = PHI

        self.A = self.PHI@self.D@self.PHI.T - self.gamma*self.PHI@self.D@self.P@self.PHI.T
        self.b = self.PHI@self.D@torch.diagonal(self.P@self.R.T).reshape(dim,1)
        self.theta_opt = torch.linalg.inv(self.A)@self.b

    def sample(self, size):
        data = []
        for _ in range(size):
            s1 = self.states[torch.multinomial(self.stat_dist, 1).item()]
            s2 = self.states[torch.multinomial(self.P[[s1], :], 1).item()]
            r = self.R[s1, s2].item()
            data.append((s1,s2,r))
        return data






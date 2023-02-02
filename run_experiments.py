import torch

from datetime import datetime

from env_sampling.mdp_env_obj import mdp_env
from env_sampling.env_solver_wrapper import env_solver

from dataset_sampling.mdp_env import build_dataset
from dataset_sampling.gym_env import build_dataset as build_dataset_env
from dataset_sampling.dataset_solver_wrapper import dataset_solver

from torch.multiprocessing import Pool
import json
import argparse

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def run_experiment_fs(inputs):
    args, solvers, exp_num = inputs
    args['dtype'] = torch.float64 if args['dtype'] == "float64" else torch.float32
    EXPERIMENT_RESULT = {'exp_num': exp_num}
    print('Starting experiment', exp_num)
    if args['type'] == "MDP":
        with torch.no_grad():
            problem_set = build_dataset(args)
    elif args['type'] == "gym":
        args['feature_type'] = args.get('feature_type', 'kernel')
        max_loop_iter = args.get('max_loop_iter', float('inf'))
        min_loop_iter = args.get('min_loop_iter', 0)
        num_iter = 0
        while num_iter > max_loop_iter or num_iter < min_loop_iter:
            problem_set = build_dataset_env(args)
            A_hat = problem_set['A_hat']
            eig_min = torch.linalg.eigvals(1 / 2 * (A_hat + A_hat.T)).real.min().item()
            num_iter = 16 / eig_min
            print(num_iter)
    shift = torch.randn(problem_set['theta_est'].shape, dtype=args['dtype'], device=device)
    shift = shift * args['shift_length'] / torch.norm(shift)
    problem_set['theta_init'] = problem_set['theta_est'] + shift

    max_iter = args.get('max_iter', None)
    for solver_args in solvers:
        result = dataset_solver(args, solver_args, problem_set)
        if args['max_iter'] is None:
            args['max_iter'] = result['milestones'][-1]
        EXPERIMENT_RESULT[solver_args['title']] = result

    return EXPERIMENT_RESULT

def run_experiment_env(inputs):
    args, solvers, exp_num = inputs
    args['dtype'] = torch.float64 if args['dtype'] == "float64" else torch.float32
    EXPERIMENT_RESULT = {'exp_num': exp_num}
    with torch.no_grad():
        print('Starting experiment', exp_num)
        if args['type'] == "MDP_iid":
            env = mdp_env(args)
        shift = torch.randn(env.theta_opt.shape,
                            dtype=args['dtype'], device=device)
        shift = args["shift_length"]*shift / torch.norm(shift)
        args['theta_init'] = env.theta_opt + shift

        max_iter = args.get('max_iter', None)
        for solver_args in solvers:
            result = env_solver(args, solver_args, env)
            if args['max_iter'] is None:
                args['max_iter'] = result['milestones'][-1]
            EXPERIMENT_RESULT[solver_args['title']] = result

    return EXPERIMENT_RESULT


def main(experiment_config):
    num_experiments = experiment_config['num_experiments']
    num_processes = experiment_config['num_parallel_processes']
    experiment_type = experiment_config['experiment_type']
    glob_args = experiment_config["experiment_arguments"]
    solvers = experiment_config["solvers"]

    if 'out_path' in experiment_config:
        out_path = experiment_config['out_path']
    else:
        now = datetime.now()
        out_path = 'Results/EXP_results_' + now.strftime("%H_%M_%S") + '.pth'
    if num_processes == 1:
        results = []
        for exp_num in range(1, num_experiments+1):
            if experiment_type == 'dataset':
                results.append(run_experiment_fs( (glob_args, solvers, exp_num) ))
            elif experiment_type == 'environment':
                results.append(run_experiment_env( (glob_args, solvers, exp_num) ))
    else:
        if experiment_type == 'dataset':
            with Pool(processes=num_processes) as p:
                results = p.map(run_experiment_fs, [(glob_args, solvers, i) for i in range(1, num_experiments+1)])
        elif experiment_type == 'environment':
            with Pool(processes=num_processes) as p:
                results = p.map(run_experiment_env, [(glob_args, solvers, i) for i in range(1, num_experiments+1)])
    torch.save(results, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment setting.')
    parser.add_argument("-p", "--path", type=str,
                        help="Path to experiment setup file")
    pargs = parser.parse_args()
    experiment_config = json.load(open(pargs.path, 'r'))
    main(experiment_config)

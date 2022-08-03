import torch
import datetime as dt
from mdp_env import build_dataset

def run_experiments(args):
    results = []
    dataset = build_dataset(args)

    for _ in args['num_of_experiments']:
        result = run_single_experiment(dataset)
        results.append(result)
    if args['save_result']:
        now = dt.datetime.now()
        if args['save_name'] is None:
            torch.save(results, 'Results/'+'_'.join([args['alg_name'], args['env_name'], now]))
        else:
            torch.save(results, 'Results/' + args['save_name'])


def run_single_experiment(dataset):
    algo
    result = 0
    return result


if __name__ == '__main__':
    args = {'env': 'mdp', 'save_results': True}
    run_experiments(args)

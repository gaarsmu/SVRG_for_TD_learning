from dataset_sampling.SVRG_solver import solve as SVRG_solve
from dataset_sampling.policy_iter_solve import solve as pi_solve
from dataset_sampling.PD_SVRG_solver import solve as PDSVRG_solve
from dataset_sampling.gtd2_solver import solve as gtd2_solve
from dataset_sampling.bSVRG_solver import solve as bSVRG_solve


def dataset_solver(args, solver_args, problem_set):
    if solver_args['type'] == 'SVRG':
        result = SVRG_solve(problem_set, args)
    elif solver_args['type'] == 'TD':
        args['learning_rate'] = solver_args['learning_rate']
        args['lr_value'] = solver_args.get('lr_value', 1.)
        args['decay_rate'] = solver_args.get('decay_rate', 0.5)
        result = pi_solve(problem_set, args)
    elif solver_args['type'] == 'batched_SVRG':
        result = bSVRG_solve(problem_set, args)
    elif solver_args['type'] == 'GTD2':
        args['learning_rate'] = solver_args['learning_rate']
        args['lr_value'] = solver_args.get('lr_value', 1.)
        args['lr_ratio'] = solver_args.get('lr_ratio', 1.)
        args['decay_rate'] = solver_args.get('decay_rate', 0.5)
        result = gtd2_solve(problem_set, args)
    elif solver_args['type'] == 'PDSVRG':
        args['learning_rate'] = solver_args['learning_rate']
        args['lr_value_w'] = solver_args['lr_value_w']
        args['lr_value_theta'] = solver_args.get('lr_value_theta', None)
        args['lr_ratio'] = solver_args.get('lr_ratio', None)
        args['batch_size'] = solver_args.get('batch_size', '16/eig_min')
        result = PDSVRG_solve(problem_set, args)
    return result
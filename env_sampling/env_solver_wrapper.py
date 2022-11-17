from env_sampling.SVRG_env_solver import solve as SVRG_env_solve
from env_sampling.policy_iter_env_solve import solve as PI_env_solve
from env_sampling.VRTD_env_solve import solve as VRTD_env_solve

def env_solver(args, solver_args, env):
    if solver_args['type'] == 'SVRG':
        args['conv_rate'] = solver_args['conv_rate']
        result = SVRG_env_solve(env, args)
    elif solver_args['type'] == 'TD':
        args['learning_rate'] = solver_args['learning_rate']
        args['lr_value'] = solver_args.get('lr_value', 1.)
        args['decay_rate'] = solver_args.get('decay_rate', 0.5)
        result = PI_env_solve(env, args)
    elif solver_args['type'] == 'VRTD':
        args["batch_size"] = solver_args["batch_size"]
        args["learning_rate"] = solver_args["learning_rate"]
        result = VRTD_env_solve(env, args)
    return result

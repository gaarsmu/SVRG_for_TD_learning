import torch


def compute_f(problem_data, theta_cur, theta_opt, args):
    A_hat = problem_data['A_hat']
    return ((theta_cur - theta_opt).T @ A_hat @ (theta_cur - theta_opt)).item()


def compute_f_env(env, theta_cur, theta_opt):
    return ((theta_cur - theta_opt).T@env.A@(theta_cur-theta_opt)).item()

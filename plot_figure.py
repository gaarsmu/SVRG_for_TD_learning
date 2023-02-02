import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime
import json

def main(experiment_config):
    exp_data = experiment_config['out_path']
    report_freq = experiment_config["experiment_arguments"]["report_freq"]
    if 'figure_path' in experiment_config:
        figure_path = experiment_config['figure_path']
    else:
        now = datetime.now()
        figure_path = 'Experiment_figure_' + now.strftime("%H_%M_%S") + '.png'

    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)

    fig.set_figheight(5)
    fig.set_figwidth(15)

    EXPERIMENT_RESULT = torch.load(exp_data)
    num_exps = len(EXPERIMENT_RESULT)
    x_tics = np.arange(0,  experiment_config["experiment_arguments"]["max_iter"]//report_freq+1)
    solver_title = [x['title'] for x in experiment_config['solvers']]
    tpls = [('#377eb8', 'o'), ('#ff7f00', '^'), ('#4daf4a', '+'),
            ('#f781bf', 'P'), ('#a65628', 'd'), ('#984ea3', 'x')]

    tpls = [(solver_title[i], tpls[i][0], tpls[i][1]) for i in range(len(solver_title))]

    x_scale_fontsize = 15
    y_scale_fontsize = 17
    title_fontsize = 20

    x_label_text = 'Number of batches of size {}'.format(report_freq)

    if experiment_config['experiment_type'] == 'dataset':
        if experiment_config["experiment_arguments"]['type'] == 'gym':
            title_text = experiment_config["experiment_arguments"]['gym_env']
        else:
            title_text = experiment_config["experiment_arguments"]['type']
    else:
        title_text = 'MDP, iid'

    for alg, color, marker in tpls:
        value_array = [np.array(EXPERIMENT_RESULT[num][alg]['distances']) for num in range(num_exps)]
        max_len = np.min([x.shape[0] for x in value_array])
        value_array = np.vstack([x[:max_len] for x in value_array])
        value_array_logs = np.log(value_array)
        means = np.exp(value_array_logs.mean(axis=0))
        upper_bound = np.exp(value_array_logs.mean(axis=0) + value_array_logs.std(axis=0))
        lower_bound = np.exp(value_array_logs.mean(axis=0) - value_array_logs.std(axis=0))
        axs[0].plot(x_tics[:max_len], means, color=color, marker=marker, label=alg)
        axs[0].fill_between(x_tics[:max_len], lower_bound, upper_bound, color=color, alpha=.2)
    axs[0].set_yscale('log')
    axs[0].set_title(title_text + ", f", fontsize=title_fontsize)
    axs[0].legend(loc="lower left")
    axs[0].set_xlabel(x_label_text, fontsize=x_scale_fontsize)
    axs[0].set_ylabel('log(f)', fontsize=y_scale_fontsize)

    for alg, color, marker in tpls:
        value_array = [np.array(EXPERIMENT_RESULT[num][alg]['norms']) for num in range(num_exps)]
        max_len = np.min([x.shape[0] for x in value_array])
        value_array = np.vstack([x[:max_len] for x in value_array])
        value_array_logs = np.log(value_array)
        means = np.exp(value_array_logs.mean(axis=0))
        upper_bound = np.exp(value_array_logs.mean(axis=0) + value_array_logs.std(axis=0))
        lower_bound = np.exp(value_array_logs.mean(axis=0) - value_array_logs.std(axis=0))
        axs[1].plot(x_tics[:max_len], means, color=color, marker=marker, label=alg)
        axs[1].fill_between(x_tics[:max_len], lower_bound, upper_bound, color=color, alpha=.2)
    axs[1].set_yscale('log')
    axs[1].set_title(title_text + ", norm", fontsize=title_fontsize)
    axs[1].legend(loc="lower left")
    axs[1].set_xlabel(x_label_text, fontsize=x_scale_fontsize)
    axs[1].set_ylabel('log(norm)', fontsize=y_scale_fontsize)
    plt.savefig(figure_path)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Experiment setting.')
    parser.add_argument("-p", "--path", type=str,
                        help="Path to experiment setup file")
    pargs = parser.parse_args()
    experiment_config = json.load(open(pargs.path, 'r'))
    main(experiment_config)

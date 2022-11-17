This is anonymous github repository for
**"Closing the gap between SVRG and TD-SVRG with Gradient Splitting"**. 
To run experiment (one dataset or iid environment) execute:

```
python run_experiment.py -p {path_to_experiment_setup_file.json}
```
To produce figure based on experiment results execute:

```
python plot_figure.py -p {path_to_experiment_setup_file.json}
```
Note that the same experiment setup file should be passed
as an argument to 
**run_experiment.py** and **plot_figure.py** scripts. Setup files 
to reproduce paper experiments might be found inside this repository.

Code is tested on Python v. 3.9 and Pytorch v. 1.11.0, but doesn't
use any specific features of these versions and should be executable
on any recent version of Python and Pytorch.

Authors run code on single NVIDIA RTX GeForce 3060 GPU with 6GB memory.
Before running the code on your machine, make sure to set number of
parallel processes ("num_parallel_processes" parameter in 
epxeriment_setup file) so that it fits to your video memory. Set 
this value to 1 to disable parallelization.
### global parameter settings ###

# random seed
seed = 0

## controller settings
# job status signals
status_submitted = 0
status_running = 1
status_stopped = 2
status_complete = 3

# waiting status
wait_for_start = 4
wait_for_resuming = 5
wait_for_scaling = 6

# parameters for max batch size control
num_gpus_per_node = 4
max_size_limit = 8
scale_down_factor = 0.5
num_epochs_before_next_scaling = 3

## scaling settings
# worker status
worker_initializing = 7
worker_running = 8
worker_scale_ready = 9
worker_scale_complete = 10
worker_complete = 11

# exit status
exit_released = 12
exit_stopped = 13
exit_complete = 14

## training settings
convergence_patience = 10
convergence_delta = 0.01
lr_scaling_factor = 1.0

## scheduler settings
num_jobs = 1

# parameters of evolutionary algorithm
population_size = 8
crossover_probability = 0.5
mutate_rate = 0.1
update_interval = 1

# monitor settings
predictor_input_size = 5
predictor_hidden_size = 64
predictor_epoch_size = 2000
predictor_batch_size = 8
predictor_train_epochs = 60
num_monitored_throughputs = 5
monitored_status_active = 15
monitored_status_complete = 16

## paths
# path to store traces
trace_path = 'traces/'
# path to store logs
log_path = 'log/dev/'
# path to store checkpoints
ckpt_path = 'checkpoints/dev/'
# path to job history
job_history_path = 'traces/job_history.csv'

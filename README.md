# ONES Scheduler for SC'21 AD/AE

We present a prototype of ONES, which is implemented with [RPyC](https://rpyc.readthedocs.io/) and [PyTorch](https://pytorch.org/).

---

## Setup

### Environment

1. [TACC Frontera RTX Nodes](https://frontera-portal.tacc.utexas.edu/user-guide/system/#gpu-nodes).
2. CUDA requirements:
   ```
   CUDA == 10.1
   cuDNN >= 7.6.3
   NCCL >= 2.4.8
   ```
3. Download the repository.
   ```
   $  git clone https://github.com/kurisusnowdeng/ones.git
   ```
4. Setup the virtual environment by runnning `scripts/env_setup.sh`. The following libraries will be installed.
   ```
   python == 3.7
   rpyc >= 5.0.1
   pytorch == 1.4.0
   torchvision == 0.5.0
   numpy >= 1.17.4
   scipy == 1.3.1
   pytorch-pretrained-bert >= 0.6.2
   ```

---

## Usage

### A Simple Example

1. Launch the system.
   
   ```
   $  python -m src.controller --size NUM_NODES --port CONTROLLER_PORT --cache-dir PATH/TO/CACHE/
   ```
   
   `--size` is compulsory to specify the number of nodes to use.
2. On each worker node, make sure that there is no irrelevant process using any GPU. Join the node to the controller.
   
   ```
   $  python -m src.app_manager --port MANAGER_PORT --controller_port CONTROLLER_PORT --cache-dir PATH/TO/CACHE/
   ```
3. Submit your job.
   
   ```
   $  python -m src.workload submit path/to/your_script.py \
      --batch-size=BATCH_SIZE --lr=LEARNING_RATE \
      --dataset-size=DATASET_SIZE --early-stop-patience=PATIENCE
   ```

### Run Experiments on TACC

1. Set `/path/to/project` in `scripts/master.slurm` and `scripts/worker.slurm` to your project directory.
2. Submit the job to the `rtx` queue.
   ```
   $  ./scripts/launch.sh -j JOB_NAME -n 16 -t 06:00:00
   ```
3. After the job is completed, extract and analyze results from logs (defaultly located in `log/test`, which can be modified in `src/config.py`).
   ```
   $  python ./scripts/measurement.py
   ```
   This script will generate the plots as presented in our paper.

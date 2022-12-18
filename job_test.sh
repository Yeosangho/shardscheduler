#!/bin/bash

#SBATCH -J pytorch_test
#SBATCH -p cas_v100_4
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4 # using 2 gpus per node
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH --comment pytorch

module purge
module load cuda/11.3 python/3.7.1
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export TRAINER_PORT=12342
export HANDLER_PORT=30001
export WORLD_SIZE=8
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_IFNAME=ib0
export MASTER_SOCKET_IFNAME=ib0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_DISABLE=0
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source activate lightning

### the command to run
#srun /home01/hpc72a03/.conda/envs/shard/bin/python main.py --sdp_ratio 0.0 --fsdp_ratio 0.0 --dp_ratio 1.0 --bucket_size 10 --target_memory 0.53
srun /home01/hpc72a03/.conda/envs/lightning/bin/python main_gpt2_with_health_checker.py --sdp_ratio 0.0 --fsdp_ratio 0.0 --dp_ratio 1.0 --bucket_size 500 --target_memory 31
#srun /home01/hpc72a03/.conda/envs/shard/bin/python run.py --python_path "/home01/hpc72a03/.conda/envs/shard/bin/python"
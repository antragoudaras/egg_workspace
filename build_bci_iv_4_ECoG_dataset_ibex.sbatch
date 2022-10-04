#!/bin/bash --login
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=32G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./building_folder_mne_data_bci_iv_4_ECoG_log_outputs/%x-%j-slurm.out
#SBATCH --error=./building_folder_mne_data_bci_iv_4_ECoG_log_outputs/%x-%j-slurm.err

set -e

conda activate ../env

python train_flexion_fingers_cropped.py
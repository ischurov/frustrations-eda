#!/bin/bash

#SBATCH --job-name="vscode"
#SBATCH --time=12:00:00     # walltime
#SBATCH -n1
#SBATCH -c16
#SBATCH -p genoa

set -x

# export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ishchurov/frustrations-eda

#nix develop .#default --builders 'ssh://cn71?remote-program=/home/ischurov/.local/bin/nix-store - - 64 2 kvm,big-parallel' --max-jobs 0 --command code tunnel --accept-server-license-terms 
nix develop .#default --command code tunnel --accept-server-license-terms 



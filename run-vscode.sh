#!/bin/bash

# Submit the job and get the job id
output=$(sbatch sbatch-vscode.sh)

echo $output

jobid=$(echo $output | awk '{print $NF}')

echo Found jobid: $jobid

# Define the name of the output file
outfile="slurm-$jobid.out"

echo Waiting for outfile

# Wait until the output file is created
while [ ! -f "$outfile" ]; do
  sleep 1
done

# Display job output
watch "tail $outfile"
s

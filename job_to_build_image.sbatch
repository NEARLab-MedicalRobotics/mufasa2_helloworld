#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --partition=jobs
#SBATCH --qos=build
#SBATCH --output=./logs/build_sbatch-%j.out
#SBATCH --error=./logs/build_sbatch-%j.err

#---Load the Singularity module
module load amd/singularity

#---Build the container image
singularity build --fakeroot --nv image_built.sif image_definition.def   
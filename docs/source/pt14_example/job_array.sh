#!/bin/bash
#SBATCH --job-name=Pt14_example
#SBATCH --partition=qtest
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH --array=1-10%10

echo "========= Job started  at `date` =========="

SOURCE_FILE=... # Python environment with AGOX installed

source $SOURCE_FILE 
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MAINDIR="$(pwd)"
SCRATCH=/scratch/$SLURM_JOB_ID/run_$SLURM_ARRAY_TASK_ID

mkdir $SCRATCH
cp Pt14_example.py $SCRATCH/.
cd $SCRATCH
python Pt14_example.py -i $SLURM_ARRAY_TASK_ID
cd $MAINDIR
cp -rf $SCRATCH/ $MAINDIR/.

echo "========= Job finished at `date` =========="


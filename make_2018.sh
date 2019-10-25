#!/bin/sh

function write_script
{
# Write the SLURM file
    STUDY_NAME=$(printf '2018')
    MULT=$( printf "mult%01d" $NMULT )
    DIR_NAME=$( printf "%s/gpu%01d/%s/" ${STUDY_NAME} $NGPU $MULT )
    mkdir -p $DIR_NAME
    echo $NGPU $MULT $DIR_NAME
cat << _EOF_ > ${DIR_NAME}/run.slurm
#!/bin/bash

#SBATCH --job-name=gpu2018-${NGPU}.${NMULT}
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:${NGPU}
#SBATCH --qos=medium+
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=MaxMemPerNode
#SBATCH --constraint=hpcf2018  # Not needed unless you want V100's
#SBATCH --exclusive

# Load CUDA
module load CUDA/9.0.176
# Need this for anaconda
export CONDA_PREFIX=${CONDA_PREFIX}
export PYTHONPATH=\${CONDA_PREFIX}/bin
export PATH=\$CONDA_PREFIX/bin:\$PATH

export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:$CONDA_PREFIX/cuDNN-7.1.4-9.0/lib64/"

# Need these because 9.0 is missing on 2018
# The CUDA toolkit location
TOOLKITDIR=$CONDA_PREFIX/cuda-toolkit9.0/
# The end-user drivers for the VXXX's GPUs


#hostname
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/umbc/xfs1/cybertrn/cybertraining2019/team3/research/results/source/libfabric/lib/"
srun \$PYTHONPATH/python ${EXE} -a False -fs -d ${DATA}/${MULT}/ -f params.json 
_EOF_

# Write the JSON file
cat << _EOF_ >${DIR_NAME}/params.json
{
    "epochs": [5,10,15],
    "learning_rate": [0.001],
    "data_multiplier": [${NMULT}],
    "gpu": [${NGPU}],
    "batch_size": [128,256,512,1024,2048,4096,8192,16384,32768,65536]
}
_EOF_
}
# Constants needed for file creation
CONDA_PREFIX=/umbc/xfs1/cybertrn/cybertraining2019/team3/research/miniconda3_gpu_master/
SRC=./
DATA=/umbc/xfs1/cybertrn/cybertraining2019/team3/research/results/data/
EXE=$SRC/main.py
for NGPU in 1 2 3 4
do
    for NMULT in 1 2 4
    do
        write_script
    done
done

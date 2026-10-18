#!/bin/bash

# Slurm job settings
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --job-name=MYJOB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<CPU_COUNT>
#SBATCH --mem=<MEMORY_AMOUNT>        
#SBATCH --time=02:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --gres=gpu:<GPU_COUNT>        # set GPUs needed

set -euo pipefail

# Application command and args
# Launcher + script should accept a checkpoint directory and an optional resume flag.
# Specify the arg names below to match trainer
CKPT_ARG_NAME="--ckpt_dir"
RESUME_ARG_NAME="--resume"
APP_CMD="<launcher_and_script>"   # ( "<launcher> <script>.py")
APP_ARGS=""                            # ("--batch_size 8 --lr 3e-5")
APP_STDIN=""
APP_STDOUT=""

# Load modules
module purge
# module load <stack>   # ex, cuda/<ver>, python/<ver>, mpi/<ver>

# Define paths and scratch space
JOB_TAG="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
SCRATCH_ROOT="/scratch/${USER}"
JOB_SCRATCH="${SCRATCH_ROOT}/${JOB_TAG}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR}"
CKPT_DIR="${JOB_SCRATCH}/ckpt"

mkdir -p "${JOB_SCRATCH}" "${CKPT_DIR}"

# Define checkpoint and cleanup behavior
checkpoint() {
  echo "checkpoint_at_$(date +%s)" > "${CKPT_DIR}/resume.token"
  sync
}
finish() {
  rsync -pru --exclude='*.tmp' "${JOB_SCRATCH}/" "${SUBMIT_DIR}/${JOB_TAG}/"
}
trap 'checkpoint; exit 0' USR1
trap 'finish' EXIT

# Resume if checkpoint exists
RESUME_FLAG=""
if compgen -G "${CKPT_DIR}/*" >/dev/null && [[ -f "${CKPT_DIR}/resume.token" ]]; then
  RESUME_FLAG="${RESUME_ARG_NAME} ${CKPT_DIR}"
elif [[ -f "${CKPT_DIR}/resume.token" ]]; then
  RESUME_FLAG="${RESUME_ARG_NAME} ${CKPT_DIR}"
fi

# Move to scratch and run job
cd "${JOB_SCRATCH}"

# Always pass the checkpoint directory to the trainer
BASE_CMD="${APP_CMD} ${CKPT_ARG_NAME} ${CKPT_DIR} ${APP_ARGS} ${RESUME_FLAG}"

if [[ -n "${APP_STDIN}" || -n "${APP_STDOUT}" ]]; then
  if [[ -n "${APP_STDIN}" && -n "${APP_STDOUT}" ]]; then
    srun ${BASE_CMD} < "${SUBMIT_DIR}/${APP_STDIN}" > "${APP_STDOUT}"
  elif [[ -n "${APP_STDIN}" ]]; then
    srun ${BASE_CMD} < "${SUBMIT_DIR}/${APP_STDIN}"
  else
    srun ${BASE_CMD} > "${APP_STDOUT}"
  fi
else
  srun ${BASE_CMD}
fi
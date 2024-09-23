module load craype-x86-genoa
module load cce/17.0.1
module load rocm/6.1.1
module load cray-mpich
module load craype-accel-amd-gfx942
module load cray-pals

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export CRAY_OMP_CHECK_AFFINITY=FALSE
export OMP_NUM_THREADS=24
export ROCR_VISIBLE_DEVICES=0
export HSA_XNACK=1
export OMP_TARGET_OFFLOAD=MANDATORY
export CRAY_ACC_DEBUG=0

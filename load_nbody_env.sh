module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

# only load these if you actually need compiled AMUSE/HDF5/CUDA later
# module load cuda
# module load hdf5

source /home/wyz5rge/.venv/nbody/bin/activate

# Project data
export PROTOCLUSTER_DATA=/standard/Tan_JC/backup_protoclusters/multiples

# SPISEA model/data paths
export PYSYN_CDBS=/home/wyz5rge/SPISEA/cdbs
export SPISEA_MODELS=/home/wyz5rge/SPISEA

# Source code paths.
# Use case blocks so repeated "source load_nbody_env.sh" calls do not duplicate PYTHONPATH entries.

# SPISEA source code
case ":$PYTHONPATH:" in
    *":/home/wyz5rge/SPISEA:"*) ;;
    *) export PYTHONPATH=/home/wyz5rge/SPISEA:$PYTHONPATH ;;
esac

# TurbulentClusterModel source code, needed for:
# from nbody62spisea import converter
case ":$PYTHONPATH:" in
    *":/home/wyz5rge/TurbulentClusterModel:"*) ;;
    *) export PYTHONPATH=/home/wyz5rge/TurbulentClusterModel:$PYTHONPATH ;;
esac

# CMD generator source code, needed for:
# import interpolator
case ":$PYTHONPATH:" in
    *":/scratch/wyz5rge/synthetic-hr/cmd_generator:"*) ;;
    *) export PYTHONPATH=/scratch/wyz5rge/synthetic-hr/cmd_generator:$PYTHONPATH ;;
esac

# Keep threaded numerical libraries from oversubscribing small jobs/login sessions.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
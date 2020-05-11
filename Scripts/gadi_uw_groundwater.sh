#PBS -P q97
#PBS -q express
#PBS -l walltime=02:00:00
#PBS -l mem=190GB
#PBS -l jobfs=100GB
#PBS -l ncpus=48
#PBS -l software=underworld
#PBS -l wd
#PBS -N uw_groundwater_heatflow
#PBS -l storage=scratch/q97

source /scratch/q97/codes/UWGeodynamics_2.9.5.sh

mpirun -np 48 python3 03-underworld-model-groundwater-heatflow.py /scratch/q97/brm563/sydney_basin/ --res 25 60 80 -v

#! /bin/bash
#PBS -N NR_Thesis
#PBS -o /home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/logs/out.log
#PBS -e /home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/logs/err.log
#PBS -l ncpus=100
#PBS -q cpu

rm -r /home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/results
rm -r /home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/logs
mkdir /home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/logs
mkdir /home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/results


source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate cv

python  /home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/get_activations.py

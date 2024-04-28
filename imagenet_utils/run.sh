#! /bin/bash
#PBS -N ImagenetUtils
#PBS -o /home/niranjan.rajesh_asp24/thesis-manifolds/imagenet_utils/logs/out.log
#PBS -e /home/niranjan.rajesh_asp24/thesis-manifolds/imagenet_utils/logs/err.log
#PBS -l ncpus=1
#PBS -q cpu


rm -r /home/niranjan.rajesh_asp24/thesis-manifolds/imagenet_utils/logs
mkdir /home/niranjan.rajesh_asp24/thesis-manifolds/imagenet_utils/logs



source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate cv

python  /home/niranjan.rajesh_asp24/thesis-manifolds/imagenet_utils/imagenet_valid_labels.py 

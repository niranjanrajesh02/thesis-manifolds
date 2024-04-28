#! /bin/bash
#PBS -N ResNet_CAMs
#PBS -o /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs/out.log
#PBS -e /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs/err.log
#PBS -l ncpus=100
#PBS -q cpu


rm -r /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs
mkdir /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs



source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate cv

python  /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/attacks.py

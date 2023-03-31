# Monograph-Image-Retrieval
The goal of this project is to use recent advances in monocular depth, instance segmentation, scene graphs, and graph neural networks to push the state-of-the-art in indoor image retrieval.


## docker in local machine
Before running docker in your local machine, make sure you have docker installed in your machine.
Once you have it installed, follow the steps to run docker for this repo:


Pull docker from dockerhub
```
docker pull rupalsaxena/3dvision_grp11:latest
```
Check if image is installed using:
```
docker images
```
If you see installed docker image, you are good to go to next step. Run the docker image.
```
sh docker/start_cpu_docker.sh
```
If it runs successfully, you are inside docker.


## docker using Singularity in ETHZ HPC euler
To use Singularity in euler, your NETHZ username has to be added to ID-HPC-SINGULARITY group.

Request a compute node with Singularity. This step will take some time. 
```
bsub -n 1 -R singularity -R light -Is bash
```

Load module eth_proxy to connect to the internet from a compute node
```
module load eth_proxy
```
Pull the Docker image with Singularity
```
cd $SCRATCH
singularity pull docker://rupalsaxena/3dvision_grp11
ls
```
Running ```ls``` above should return:
3dvision_grp11_latest.sif

Run the container as follows. Note: change euler_username to your euler username.
```
singularity run --bind /cluster/home/euler_username:/home 3dvision_grp11_latest.sif 
```
Keep this repo in /cluster/home/euler_username path so that your repo can be mounted automatically inside docker. 

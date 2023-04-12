# Monograph-Image-Retrieval
The goal of this project is to use recent advances in monocular depth, instance segmentation, scene graphs, and graph neural networks to push the state-of-the-art in indoor image retrieval.


## Docker in local machine
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
If you see installed docker image, you are good to go to next step. 

***Using visual studio code or any IDE, open docker/start_cpu_docker.sh file. -v path1:path2 in docker script means that you are mounting path1 in your local machine to path2 inside the docker. Make sure you correctly mount this repository and dataset inside the docker. Path of this repo is correct, however path of dataset needs to be changed depending on where you are storing your data in your local machine. Once mounting is correctly mentioned in the file, go ahead and run the docker.***

Run the docker image using following command
```
sh docker/start_cpu_docker.sh
```
If it runs successfully, you are inside docker.


## Docker using Singularity in ETHZ HPC euler
To use Singularity in euler, your NETHZ username has to be added to ID-HPC-SINGULARITY group.

Request a compute node with Singularity. This step will take some time. 
```
srun --pty --mem-per-cpu=4G --gpus=1 bash
```

Load module eth_proxy to connect to the internet from a compute node
```
module load eth_proxy
```
Navigate to $SCRATCH
```
cd $SCRATCH
```
Pull the Docker image with Singularity. ***Please note that if you have already pulled a docker in past, you can skip this step.***
```
singularity pull docker://rupalsaxena/3dvision_grp11
```
Check if singularity file is available
```
ls 
```
Running ```ls``` above should return:


***3dvision_grp11_latest.sif***

If this file is available, you are good to go. Otherwise, contact maintainer of docker container of this repo. 


Run the container as follows. ***Note: euler_username = your euler username***
```
singularity run --bind /cluster/home/euler_username:/home --bind /cluster/project/infk/courses/252-0579-00L/group11_2023/datasets:/mnt/datasets 3dvision_grp11_latest.sif 
```
Keep this repo in /cluster/home/euler_username path so that your repo can be mounted automatically inside docker.

Once you run this, you are inside the singularity docker. 
1. Navigate to home folder to see this repo inside the docker.
2. Navigate to mnt folder to see the data inside the docker.

If you can see both the folders mentioned above, congratulations, you are good to go!

## Run pipeline

Please follow the steps to run the pipeline.

***Step 1:*** Make sure you have hypersim data installed. In euler, this dataset is already stored in shared repo provided by 3d vision course.

***Step 2:*** Make sure you have this repository cloned. In euler, keep this repo in your home folder. 

***Step 3:*** Make sure you have docker or Singularity installed 

***Step 4:*** Run the docker image using the method mentioned above in this readme file. ***While running docker or Singularity make sure that you mount this repo and the path of the hypersim data inside the docker correctly.***

***Step 5:*** Navigate to ```src/Monograph/dataloader/hypersim_config.py``` file of this repo and update HYPERSIM_DATAPATH value to path of the hypersim data. Please note that HYPERSIM_DATAPATH should be path inside docker of your data.

***Step 6:*** Use the following commands to save graphs from hypersim dataset.

```
cd src/Monograph/
python3 main_save_graphs.py
```
***Step 7:*** Sit back and enjoy because running this is gonna take a while :)


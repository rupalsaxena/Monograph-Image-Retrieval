# Monograph-Image-Retrieval

The goal of this project was to use recent advances in monocular depth, instance segmentation, scene graphs, and graph neural networks to push the state-of-the-art in indoor image retrieval.

## Setup environment in euler cluster of ETHZ 
Clone this public repo: 
```
git clone https://github.com/rupalsaxena/Monograph-Image-Retrieval.git
```
TODO: add virtual env in package, how to run virtual environment, ask for space using slurm or something and then run the project


## Pipeline overview
Shown below is the pipeline overview of this project. 

![Alt text](https://github.com/rupalsaxena/Monograph-Image-Retrieval/blob/final_code/images/pipeline.png)

We will see how to run each element of this pipeline one by one.

## Hypersim Data
The Hypersim dataset, developed by Apple Inc., consists
of a collection of photorealistic synthetic images depicting
indoor scenes, accompanied by per-pixel ground truth labels. This data consists of RGB, Semantic, and Depth instances.

Download the data from the link: [HypersimData](https://github.com/apple/ml-hypersim)

## RGB to Semantic Instances
**STEP 1**: Save rgb and groundtruth semantic instances in torch format using dataloader
```
# navigate to hypersim dataloader
cd src/Monograph/dataloader/hypersim_pytorch
```
In your IDE, update the config parameters, specially PURPOSE="Semantic"
Run the following commands to save hypersim data in pytorch format:
```
# save hypersim data in torch format
python3 save_hypersim_dataset.py
```
Data will be saved in your provided output path in torch format.

**STEP 2**: Train DeepLabv3 Resnet50 Model using transfer learning
```

```

**STEP 3**: Test and save the predicted semantic data
```

```

## RGB to Depth 
Step 1: Load and save rgb and groundtruth depth in torch format using dataloader
```

```

Step 2: Predict Depth from Rgb using AdelaiDepth pretrained network
```

```

Step 3: Test and save predicted depth data 
```

```

## Generate Scene Graph
```

```

## Train GCN network with Triplet loss using Generated Scene Graphs
```

```

## Proximity Matching
```

```























<!---



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

## Running Data Loaders

***3DSSG Data Loader*** 
To load the triplet dataset, first import the file ```src/Monograph/dataloader/3dssg/graphloader.py```. Then, instantiate the class by providing it a path to the files which contain the graphs. Finally, call the member function ```load_triplet_dataset()``` with the start and stop indices of the graphs you would like to use.

```
import sys
sys.path.append('<path from current file to graphloader.py>')
from graphloader import ssg_graph_loader

loader = ssg_graph_loader(path='<path defaults to data location on euler>')
triplet_dataloader = loader.load_triplet_dataset(start, stop, batch_size=1, shuffle=False, nyu=True, eig=True, rio=True, g_id=False, ply=True)
```

***Pipeline Data Loader*** 
To load the triplet dataset, first import the file ```src/Monograph/dataloader/pipeline_graphloader.py```. Then, instantiate the class by providing it a path to the files which contain the graphs. Finally, call the member function ```load_triplet_dataset()``` with the start and stop indices of the graphs you would like to use.

```
import sys
sys.path.append('<path from current file to pipeline_graphloader.py>')
from pipeline_graphloader import graph_loader

loader = graph_loader(path='<path defaults to data location on euler>')
triplet_dataloader = loader.load_triplet_dataset(start, stop, batch_size=1)
```
-->
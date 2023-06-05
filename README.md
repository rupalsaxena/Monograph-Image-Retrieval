# Monograph-Image-Retrieval

The goal of this project was to use recent advances in monocular depth, instance segmentation, scene graphs, and graph neural networks to push the state-of-the-art in indoor image retrieval.

## Repository Overview

```
- data
    - hypersim_train_graphs_GT    # example graphs for training
    - hypersim_test_graphs_GT     # example graphs for testing
- docker                          # docker build and docker run files
- src
    - Monograph
        - configs                 # general configs
        - dataloader              # dataloaders for different purposes
        - GNN                     # load graphs and train GCN model
        - generate_scene_graph    # generate scene graphs from ground truth data
        - preds_to_graphs         # generate scene graphs from predicted data
        - rgb-to-depth            # rgb to depth prediction
        - rgb-to-semantics        # rgb to semantic training and prediction
```

List of packages used in this project can be found in file [requirements.txt](https://github.com/rupalsaxena/Monograph-Image-Retrieval/blob/main/docker/gpu-docker/requirements.txt)


## Setting up Infrastructure using Singularity in ETHZ HPC euler
We are using docker container so that all the users can have same working environment. You can either use Singularity in ETHZ HPC euler or directly use docker container in your local machine. 

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

Pull the Docker image with Singularity.
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
singularity run --bind /cluster/home/euler_username:/cluster/home/euler_username --bind /cluster/project/infk/courses/252-0579-00L/group11_2023:/cluster/project/infk/courses/252-0579-00L/group11_2023 3dvision_grp11_latest.sif 
```
Keep this repo in /cluster/home/euler_username path so that your repo can be mounted automatically inside docker.

Once you run this, you are inside the singularity docker. 
1. Navigate to /cluster/home/euler_username folder to see this repo inside the docker.
2. Navigate to /cluster/project/infk/courses/252-0579-00L/group11_2023 folder to see the data inside the docker.

If you can see both the folders mentioned above, congratulations, your infrastructure is ready!

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
**STEP 1**: Save rgb and groundtruth semantic instances in torch format using dataloader.
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

**STEP 2**: Train DeepLabv3 Resnet50 Model using transfer learning.
```
# navigate to rgb-to-semantics directory
cd ../../rgb-to-semantics 
```
Update input path and trained model output path in config.py
Train the model
```
python3 transfer_learning.py
```
Once training is over, check the provided output path to see trained model.


**STEP 3**: Save predicted semantic data.
Update MODELPATH TESTDATAPATH, main_path in test_model.py file. Finally, run the file to get predicted semantic instances.
```
python3 test_model.py
```
Check the predicted semantic images in main_path provided in config.py file.

## RGB to Depth 
Predict Depth from Rgb using AdelaiDepth pretrained network. 

navigate to rgb-to-depth directory:
```
cd ../rgb-to-depth 
```

Update image_dir and image_dir_out folders in test_depth.py before running it. Once updated, run the following:
```
python3 test_depth.py 
cd ../../../
```
Predicted depth data should be available on image_dir_out path mentioned above.

## Generate Scene Graph
**Method 1:** Graphs from Ground Truth Images

To generate graphs from ground truth depth and ground truth semantic instances of hypersim data, do as follows:

Step 1: Update HYPERSIM_DATAPATH and HYPERSIM_GRAPHS path in "src/Monograph/generate_scene_graph/config.py" file. 

Step 2: Run graph generation using following commands:
```
cd src/Monograph
python3 main_save_graphs.py
```

**Method 2:** Graphs from predicted images

To generate graphs from predicted depth and predicted semantic instances of hypersim data, do as follows:

Step 1: Update src/Monograph/preds_to_graphs/config.py file. Make sure to give correct paths of hypersim rgb, predicted semantics, predicted depth, and output path. 

Step 2: Generate graphs from predicted data
```
cd src/Monograph/preds_to_graphs
python3 preds_to_graphs.py
```

## Train GCN network with Triplet loss using Generated Scene Graphs
To train the model from groundtruth generated graphs, run the following commands:
```
# navigate to GNN directory
cd src/Monograph/GNN

# run ground truth graph GCN training with threshold 2
python3 LoadAndTrain.py 2 ../../../data/hypersim_train_graphs_GT/
```

## Proximity Matching
To perform image retrival on example test data with different thresholds, run the following commands:
```
# navigate to GNN directory
cd src/Monograph/GNN

# run proximity matching
python3 PipelineFeatures.py
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
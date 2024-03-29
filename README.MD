# FEWSAM Few-shot Segmentation tool based on Segment Anything

**[Segment Anything](https://github.com/facebookresearch/segment-anything)**


## Installation

SAM requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Before you start installation, create an environment first:
```
conda create --name sam python==3.9
```

Install Segment Anything:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:rootvisionai/segment-anything.git
cd segment-anything; pip install -e .
```

The following dependencies are necessary for the FEWSAM:

```
pip install opencv-python PyYAML PySimpleGUI
```

Now download the model checkpoints:

More accurate <<< [VIT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 
| [VIT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) 
| [VIT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) >>> Faster

## START SERVER

Before you start the application, create a folder to put your
support images that will be used to learn from, then create a 
folder to put your query images that are going to be labeled.
Put the relative path to the folders to support_dir and query_dir in config.yml.

#### To create request json that will be sent to server
```commandline
python interface.py  
```
then adjust make_request.py according to your images and paths

Finally, run the server ...
```
python backend/server.py
```

and make request while server.py is running
```commandline
python make_request.py
```

## DOCKERIZATION
Build docker image
```
docker build -t few_shot_sam:0.0.1 .
```
Run docker container
```
docker run --gpus all -p 80:8080 -it few_shot_sam:0.0.1
```



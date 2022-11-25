# Deep-HdrReconstruction
Official [PyTorch](https://pytorch.org/) implementation of "Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss" (SIGGRAPH 2020) [Project](https://people.engr.tamu.edu/nimak/Papers/SIGGRAPH2020_HDR) | [Paper](https://people.engr.tamu.edu/nimak/Data/SIGGRAPH20_HDR.pdf)

We propose a novel deep learning approach to reconstruct an HDR image by recovering the saturated pixels of a single input LDR image in a visually pleasing way. Our method can reconstruct regions with high luminance, such as the bright highlights of the windows (red inset), and generate visually pleasing textures and details (green insert). For more information on the method please see the [project website](https://people.engr.tamu.edu/nimak/Papers/SIGGRAPH2020_HDR).

![image](https://people.engr.tamu.edu/nimak/Papers/SIGGRAPH2020_HDR/files/teaser.png)

## Requirements
This codebase was developed and tested with PyTorch 1.2 and Python 3.6.

- Python 3.6+
- Pytorch 1.2
- torchvision
- OpenCV
- Numpy
- tensorboardX
- tqdm
- Pillow
- pyexr
- OpenEXR
- scipy

```
pip install -r requirements.txt
```

You may have to install OpenEXR through the appropriate package manager before pip install (e.g. sudo apt-get install openexr and libopenexr-dev on Ubuntu).

Download the repository

```
https://github.com/marcelsan/Deep-HdrReconstruction.git
```

### Testing (on external images, for cuda-10.0)

Installation
```
conda create -n DeepHDR python=3.6
conda activate DeepHDR
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip3 install opencv-python tqdm scipy
# Download pretrained model & store into ./checkpoints
```

Running
```
export LD_LIBRARY_PATH=/data2/shaun/cuda-10.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3
python3 test_hdr_demo.py --input_dir path_to_input_folder --output_dir path_to_output_folder
```

- Note that this method outputs H and gamma in the same folder, need to do further manipulation to store in different folders if you want to compare with comparison tool

## Usage

### Pretrained model

The pretrained model checkpoints can be found in the checkpoints folder on [Google Drive](https://drive.google.com/file/d/14pvaYHS1_tlu_xhhr9xrUNcXHIZVSdII/view?usp=sharing).

### Inference

```
CUDA_VISIBLE_DEVICES=1 python test_hdr.py --test_dir <images/dir> --out_dir <out/dir> --weights <weight/path>.pth 
```

Parameters and their description:

>```test_dir```: input images directory. A few images are avaible on the data/ folder.<br/>
>```out_dir```: path to output directory.<br/>
>```weights```: path to the trained CNN weights.<br/>
<br/>

If cuda is available, it will be used. In case you want to run the model on cpu, use ```--cpu``` when executing test_hdr.py

### Jupyter Notebook

We also provide a [Jupyter Notebook](https://github.com/marcelsan/Deep-HdrReconstruction/blob/master/hdr_reconstruction.ipynb) that guides you through the steps for running the HDR reconstruction model on animage. Open the notebook with the following command:

```
jupyter notebook hdr_reconstruction.ipynb
```

Now a web-browser window will open automatically and load the Jupyter notebook. Follow the steps in order to run the model with your own data.

## Viewing HDR Images

To visualize HDR images you can use [tev](https://github.com/Tom94/tev), which allows loading several HDR file formats and is compatible with Windows, Mac and Linux. There is also a straightforward online viewer at [openhdr.org](openhdr.org).

## References
If you find this work useful for your research, please cite:

```
@article{Marcel:2020:LDRHDR,
author = {Santos, Marcel Santana and Tsang, Ren and Khademi Kalantari, Nima},
title = {Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss},
journal = {ACM Transactions on Graphics},
volume = {39},
number = {4},
year = {2020},
month = {7},
doi = {10.1145/3386569.3392403}
}
```

## Contact

Please contact Marcel Santos (mss8@cin.ufpe.br) if there are any issues/comments/questions.

## License

Copyright (c) 2020, Marcel Santana. 

All rights reserved.

The code is distributed under a BSD license. See LICENSE for information.

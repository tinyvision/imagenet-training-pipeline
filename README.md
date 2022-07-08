## Introduction

This repository is the training pipeline specific for Light-NAS Quantization models, which is easy to use. User needs to move the `nas/models` folder of Light-NAS to `model_zoo` to complete this pipeline. 

## Installation

### Prerequisites
* Linux
* Python 3.6+
* PyTorch 1.4+
* CUDA 10.0+

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n light-nas python=3.6 -y
    conda activate light-nas
    ```

2. Install torch and torchvision with the following command or [offcial instruction](https://pytorch.org/get-started/locally/).
    ```shell
    pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    if meet `"Not be found for jpeg"`, please install the libjpeg for the system.
    ```shell
    sudo yum install libjpeg # for centos
    sudo apt install libjpeg-dev # for ubuntu
    ```

3. Install other packages with the following command.

    ```shell
    pip install -r requirements.txt
    ```

***
## Easy to use

* **Train low-precision models**
    
    ```shell
    cd scripts
    sh run_train_base_best_low_aug.sh
    ```

## Main Contributors

Hesen Chen, [Zhenhong Sun](https://sites.google.com/view/sunzhenhong).
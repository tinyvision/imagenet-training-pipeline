## Introduction

This repository is the training pipeline specific for Light-NAS Quantization models, which is easy to use. User needs to move the `nas/models` folder of Light-NAS to `model_zoo` to complete this pipeline. 

Light-NAS is a utral fast training-free neural architecture search toolbox. It supports recognition, detection and mix-precision quantization search tasks without GPU requirments. You can find more information about Light-NAS at https://github.com/alibaba/lightweight-neural-architecture-search

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
***
## Results and Models

|Backbone|Param (MB)|BitOps (G)|ImageNet TOP1|Structure|Download|
|:----|:----|:----|:----|:----|:----|
|MBV2-8bit|3.4|19.2|71.90%| -| -|
|MBV2-4bit|2.3|7|68.90%| -|- |
|Mixed19d2G|3.2|18.8|74.80%|[txt](scripts/quant/mixed19d2G.txt)|[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-7d0G/quant_238_70.7660.pth.tar) |
|Mixed7d0G|2.2|6.9|70.80%|[txt](scripts/quant/mixed7d0G.txt) |[model](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/LightNAS/quant/mixed-19d2G/quant_237_74.8180.pth.tar) |

***
## Citation

If you use this toolbox in your research, please cite the paper.
```
@article{qescore,
  title     = {Entropy-Driven Mixed-Precision Quantization for Deep Network Design on IoT Devices},
  author    = {Zhenhong Sun and Ce Ge and Junyan Wang and Ming Lin and Hesen Chen and Hao Li and Xiuyu Sun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
}
```
***
## Main Contributors

Hesen Chen, [Zhenhong Sun](https://sites.google.com/view/sunzhenhong).

# Contents

[TOC]

# EQNAS: Evolutionary Quantum Neural Architecture Search for Image Classification

This repository contains the code implementation for the paper "EQNAS: Evolutionary Quantum Neural Architecture Search for Image Classification", [Paper Link](https://www.sciencedirect.com/science/article/pii/S0893608023005348)

# Abstract

EQNAS proposes a neural network structure search method based on quantum evolutionary algorithm for quantum neural networks based on quantum circuits. In the EQNAS method, by searching for the optimal network structure, the model accuracy is improved, the complexity of quantum circuits is reduced, and the burden of constructing actual quantum circuits is reduced, which is used to solve image classification tasks.

# Architecture

![Qnn Structure](https://gitee.com/Pcyslist/cartographic-bed/raw/master/mnist_qnn.png)

This model designs and implements a quantum neural network for image classification, and conducts neural architecture search based on quantum evolutionary algorithms. The network structure mainly includes two modules:

- Quantum encoding circuit Encoder: Encode different dataset images using 01 encoding and Rx encoding, respectively
- Ansatz training circuit: A two-layer quantum neural network Ansatz was constructed using double bit quantum gates (*XX* gate, *YY* gate, *ZZ* gate) and quantum *I* gate

By performing the Pauli z-operator on the output of the quantum neural network to measure the Hamiltonian expectation, and using quantum evolution algorithms to search for the architecture of the aforementioned quantum neural network, the model accuracy is improved and the circuit complexity is reduced.

# Dataset

- Dataset [MNIST](http://yann.lecun.com/exdb/mnist/) The MNIST dataset contains a total of 70000 images, of which 60000 are for training and 10000 are for testing. Each image is composed of handwritten digit images ranging from 0 to 9, with a size of 28 x 28. Each image is in the form of black background and white text, with a black background represented by 0 and white text represented by floating-point numbers between 0 and 1. The closer to 1, the whiter the color. This model filters out the "3" and "6" categories and performs binary classification.

- -Dataset [Warship](https://gitee.com/Pcyslist/mqnn/blob/master/warship.zip) To verify the classification performance of QNN on more complex image datasets and the effectiveness of our proposed EQNAS method, we used a set of ship target datasets. This dataset is a sailing ship captured by drones from different angles. The image is in JPG format with a resolution of 640 x 512. It contains two categories: Burke and Nimitz. The number of training sets for this dataset is 411 (202 Burke class and 209 Nimitz class), and the number of test sets is 150 (78 Burke class and 72 Nimitz class).

  After downloading, extract the dataset to the following directory:

  ```python
~/path/to/EQNAS/dataset/mnist
~/path/to/EQNAS/dataset/warship
  ```

# Environmental requirements

- Hardware (GPU)

    - Use GPU to build hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
    - [MinQuantum](https://www.mindspore.cn/mindquantum/docs/en/r0.7/mindquantum_install.html)

- Installation of other third-party libraries

  ```bash
  cd EQNAS
  conda env create -f eqnas.yaml
  conda install --name eqnas --file condalist.txt
  pip install -r requirements.txt
  ```

- To view details, please refer to the following resources:
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindQuantum教程](https://www.mindspore.cn/mindquantum/docs/en/r0.7/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# Quick Start

- After installing MindSpore and MindQuantum through the official website, you can follow the following steps for training and evaluation:

- GPU environment

  ```bash
  # train
  # mnist Dataset training example
  python eqnas.py --data-type mnist --data-path ./dataset/mnist/ --batch 32 --epoch 3 --final 10 | tee mnist_train.log
  OR
  bash run_train.sh mnist /abs_path/to/dataset/mnist/ 32 3 10
  # warship Dataset training example
  python eqnas.py --data-type warship --data-path ./dataset/warship/ --batch 10 --epoch 10 --final 20 | tee warship_train.log
  OR
  bash run_train.sh warship /abs_path/to/dataset/warship/ 10 10 20
  
  # Evaluation can be performed after training is completed
  # mnist Dataset evaluation
  python eval.py --data-type mnist --data-path ./dataset/mnist/ --ckpt-path /abs_path/to/best_ckpt/ | tee mnist_eval.log
  OR
  bash run_eval.sh mnist /abs_path/to/dataset/mnist/ /abs_path/to/best_ckpt/
  # warship dataset evaluation
  python eval.py --data-type warship --data-path ./dataset/warship/ --ckpt-path /abs_path/to/best_ckpt/ | tee warship_eval.log
  OR
  bash run_eval.sh warship /abs_path/to/dataset/warship/ /abs_path/to/best_ckpt/
  ```

# Script Description

## Script and code architecture

```bash
├── EQNAS
    ├── condalist.txt                   # Anaconda env list
    ├── eqnas.py                        # 训练脚本
    ├── eqnas.yaml                      # Anaconda 环境配置
    ├── eval.py                         # 评估脚本
    ├── README.md                       # EQNAS模型相关说明
    ├── requirements.txt                # pip 包依赖
    ├── scripts
    │   ├── run_eval.sh                 # 评估shell脚本
    │   └── run_train.sh                # 训练shell脚本
    └── src
        ├── dataset.py                  # 数据集生成
        ├── loss.py                     # 模型损失函数
        ├── metrics.py                  # 模型评价指标
        ├── model
        │   └── common.py               # Qnn量子神经网络创建
        ├── qea.py                      # 量子进化算法
        └── utils
            ├── config.py               # 模型参数配置文件
            ├── data_preprocess.py      # 数据预处理
            ├── logger.py               # 日志构造器
            └── train_utils.py          # 模型训练定义
```

## Script Parameters

In config.py, quantum evolutionary algorithm parameters, training parameters, dataset, and evaluation parameters can be configured simultaneously.

  ```python
  cfg = EasyDict()
  cfg.LOG_NAME = "logger"data_preprocess

  # Quantum evolution algorithm parameters
  cfg.QEA = EasyDict()
  cfg.QEA.fitness_best = []  # The best fitness of each generation

  # Various parameters of the population
  cfg.QEA.Genome = 64  # Chromosome length
  cfg.QEA.N = 10  # Population size
  cfg.QEA.generation_max = 50  # Population Iterations

  # Dataset parameters
  cfg.DATASET = EasyDict()
  cfg.DATASET.type = "mnist"  # mnist or warship
  cfg.DATASET.path = "./dataset/"+cfg.DATASET.type+"/"  # ./dataset/mnist/ or ./dataset/warship/
  cfg.DATASET.THRESHOLD = 0.5

  # Training parameters
  cfg.TRAIN = EasyDict()
  cfg.TRAIN.EPOCHS = 3  # 10 for warship
  cfg.TRAIN.EPOCHS_FINAL = 10  # 20 for warship
  cfg.TRAIN.BATCH_SIZE = 32  # 10 for warship
  cfg.TRAIN.learning_rate = 0.001
  cfg.TRAIN.checkpoint_path = "./weights/"+cfg.DATASET.type+"/final/"
  ```

For more configuration details, please refer to the `config.py` file in the `utils` directory.

## Training Process

### Train

- Running training mnist dataset in GPU environment

  When running the following command, please move the dataset to the `dataset` folder under the root directory of EQNAS. Relative paths can be used to describe the location of the dataset. Otherwise, please set the `--data-path` to an absolute path.

  ```bash
  python eqnas.py --data-type mnist --data-path ./dataset/mnist/ --batch 32 --epoch 3 --final 10 | tee mnist_train.log
  OR
  bash run_train.sh mnist /abs_path/to/dataset/mnist/ 32 3 10
  ```

  The above Python command will run in the background. You can use the `mnist_train. log` file in the current directory or/ View the results of the log files under the `.log/` directory.

  训练结束后，您可在`eqnas.py`脚本所在目录下的`./weights/`目录下找到架构搜索过程中每一个模型对应的`best.ckpt、init.ckpt、latest.ckpt`文件以及`model.arch`模型架构文件。

  After the training is completed, you can find the `eqnas. py` script in the directory where it is located Find the `best.ckpt, init.ckpt, latest.ckpt` files and the `model.arch` model architecture files corresponding to each model during the architecture search process in the `weights/` directory.

- Training Warship Dataset in GPU Environment

  ```bash
  python eqnas.py --data-type warship --data-path ./dataset/warship/ --batch 10 --epoch 10 --final 20 | tee warship_train.log
  OR
  bash run_train.sh warship /abs_path/to/dataset/warship/ 10 10 20
  ```

  View the model training results in the same way as the training results on the mnist dataset.

## Evaluation Process

### Evaluation

- Running an evaluation mnist dataset in a GPU environment

- Before running the following command, please move the dataset to the 'dataset' folder in the root directory of EQNAS. Relative paths can be used to describe the location of the dataset. Otherwise, please provide the absolute path of the `dataset`.

- Please use the checkpoint path for evaluation. Please set the checkpoint path to an absolute path.

  ```bash
  python eval.py --data-type mnist --data-path ./dataset/mnist/ --ckpt-path /abs_path/to/best_ckpt/ | tee mnist_eval.log
  OR
  bash run_eval.sh mnist /abs_path/to/dataset/mnist/ /abs_path/to/best_ckpt/
  ```

  The above Python command will run in the background, and you can view the results through the `mnist_eval.log` file.

- Running and evaluating the Warship dataset in a GPU environment

  Please refer to the evaluation of the mnist dataset.

## Export process

### Export to MindIR

- The quantum model created based on MindQuantum is currently not officially supported for export to this format
- 但为了能够将量子线路进行保存，本项目中利用Python自带pickle数据序列化包，将架构搜索得到的每一个量子模型都保存为`./weights/model/model.arch`，您可以按照`eval.py`中的方法加载模型架构
- But in order to save quantum circuits, in this project, Python's built-in pickle data serialization package is used to save every quantum model obtained from architecture search as`/weights/model/model.arch`, you can load the model architecture according to the method in`eval. py`

# Model Description

## performance

### Training performance

#### Training EQNAS on datasets warship & mnist 

| parameter           | GPU                                           | GPU                                           |
| ------------------- | --------------------------------------------- | --------------------------------------------- |
| Model version       | EQNAS                                         | EQNAS                                         |
| resource            | NVIDIA GeForce RTX 3090 ； ubuntu20.04        | NVIDIA GeForce RTX2080Ti ; ubuntu18.04        |
| Upload date         | 2022-12-6                                     | 2022-12-6                                     |
| MindSpore version   | 1.8.1                                         | 1.8.1                                         |
| MindQuantum version | 0.7.0                                         | 0.7.0                                         |
| Dataset             | warship                                       | mnist                                         |
| Training parameters | epoch=20, steps per epoch=41, batch_size = 10 | epoch=10.steps per epoch=116, batch_size = 32 |
| optimizer           | Adam                                          | Adam                                          |
| loss function       | Binary  CrossEntropy Loss                     | Binary  CrossEntropy Loss                     |
| output              | accuracy                                      | accuracy                                      |
| accuracy            | 84.0%                                         | 98.9%                                         |
| Training duration   | 7h19m29s                                      | 27h27m23s                                     |
| speed               | 631ms/step                                    | 2734ms/step                                   |

# Random situation description

- In the script `dataset.py`, when creating the ship data loader and shuffling the ship data, a random number seed was set
- To ensure the randomness of mutation and crossover operations in quantum evolutionary algorithms, after setting the random number seed mentioned above, the random number seed was immediately reset at system time

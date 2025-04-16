### 1. Introduction

IMATAC is a deep hierarchical network with denoising autoencoder designed for the imputation of high-dimensional sparse scATAC-seq data. 

### 2. Installation 

#### 2.1 OS 
  - Ubuntu 20.04

#### 2.2 Required Python Packages

Make sure all the packages listed in requirements.txt are installed.
  
  - anndata==0.11.1
  - auto_mix_prep==0.2.0
  - numpy==1.21.4
  - pandas==2.0.3
  - scikit_learn==1.6.0
  - torch==1.10.0+cu113
  - tqdm==4.61.2

#### 2.3 Install from Github

First, download the IMATAC package.

```bash
git clone https://github.com/lhqxinghun/IMATAC
```

Second, install the package with the following command.

```bash
cd ./IMATAC
pip install .
```

### 3. Example

IMATAC can be used by the fllowing steps:

```bash
python run.py config THE PATH OF YOUR CONFIG FILES \
              dataset THE PATH OF YOUR DATASET \
              epoch EPOCH \
              mark IDENTIFIER FOR THE OUTPUT FILES \
              num_components NUMBER OF COMPONENTS FOR THE CLASSIFIER 
```

The output files will be stored in the output directory with a format of txt files.

### 4. Contact

hongqianglv@mail.xjtu.edu.cn OR leeyao@stu.xjtu.edu.cn
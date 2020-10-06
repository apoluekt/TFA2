# Installation 

## Installation using Conda 

The most straightforward way to install the TensorFlowAnalysis and AmpliTF packages is using Conda, which will handle all dependencies. Please follow the steps below: 

### Download conda
```
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
sh Miniconda2-latest-Linux-x86_64.sh
```
Answer "yes" to initialise. Log out and log in again.

### Download git packages 
```
conda install git  # if git is not installed
git clone https://github.com/apoluekt/TFA2.git
git clone https://github.com/apoluekt/AmpliTF.git
```

### Set up dependencies
```
cd TFA2/
conda env create -f env.yml  # or env_gpu.yml if you have GPU
```

### Activate TFA environment
```
conda activate tfa
```

### Configure TFA2 and AmpliTF packages
```
cd ../AmpliTF
python setup.py build
python setup.py install
cd ../TFA2
python setup.py build
python setup.py install
```

# Installation 

The most straightforward way to install the `TensorFlowAnalysis` and `AmpliTF` packages is using `Conda`. It will install an independent Python environment in your home directory (your system-wide environment is not affected) and will handle all the dependencies. Note that you will need plenty of space in your home for `Conda` environment (around 6 Gb). Please follow the steps below: 

### Download `conda`
```
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
sh Miniconda2-latest-Linux-x86_64.sh
```
Answer "yes" to initialise. Log out and log in again to activate `conda`.

### Download `git` packages 
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
This will download and install all dependent packages, inpuding TensorFlow. Be patient! 

### Activate `TFA` environment
```
conda activate tfa
```

### Configure `TFA2` and `AmpliTF` packages
```
cd ../AmpliTF
python setup.py build
python setup.py install
cd ../TFA2
python setup.py build
python setup.py install
```
### Install `ROOT` and other packages (_optional_)

The example scripts we will run do not need `ROOT`. They use `matplotlib` for plotting, `uproot` for storage and `iminuit` for minimisation. However, if you still want to stay in the `ROOT` ecosystem, and `ROOT` is not installed on your machine, you can add it from `Conda` repository `conda-forge` (in the activated `tfa` environment): 

```
conda install -c conda-forge root
```
This will add another 2 Gb to your `conda` environment. 

If you want to use `rootpy` and/or `root-numpy` as an interface between `numpy` and `ROOT`, these have to be installed with `pip`: 
```
pip install --user rootpy
pip install --user root-numpy
```

### Remove Conda environment

Once you finished playing with TensorFlow and don't want to continue, you can free up all the space it has taken by removing the `Conda` environment. `Conda` installs all dependencies in your home directory without touching the system-wide environment. To remove the `tfa` environment, do
```
conda env remove --name tfa
```

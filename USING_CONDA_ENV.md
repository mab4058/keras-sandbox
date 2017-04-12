# How to use a Conda virtual environment to play with Keras 2 on Python 3.5

### Create a virtual environement
1. `conda create -n yourenvname python=x.x`
    * Enter the name you would like where "yourenvname" appears.
	* Enter the python version you would like where "x.x" appears.
	    * I used 3.5.3
	* This should automatically install pip and some other manager packages.
        * When your environment is active, pip will install to the environment.
	
## Activate and Deactivate
### Activate
1. `source activate yourenvname`

### Deactivate
1. `source deactivate`

## Installing Packages to Environment
1. `source activate yourenvname`
2. `conda install -n yourenvname numpy scipy h5py pillow`
    * Make sure to specify the -n input parameter to install to your environment and not globally. 
2. `pip install tensorflow keras`
	* Or tensorflow-gpu.

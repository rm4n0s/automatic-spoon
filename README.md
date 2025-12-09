# Automatic Spoon
Automatic Spoon is an open source REST API to start stable diffussion's image generators and send a batch of images to create.

## This project have been postponed because of incomplete knowledge on IP Adapters. I will move to ComfyUI

## Installation
Make sure you running python 3.12. <br/>
If you are not sure, then install python 3.12 from pyenv. <br/>
When you have installed python 3.12, then call these command lines to create the local environment of the project <br/>
```bash 
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install first torch for your GPU. <br/>
There are two scripts, one to install torch for CUDA
```bash
bash install-torch-cuda.sh
```
and the other to install torch for AMD's GPU
```bash
bash install-torch-rocm.sh
```
After installing torch, run this command line to see if torch utilizes GPU successfully if it says True.  
```bash 
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available() if torch.cuda.is_available() else "CPU only")'
```
Now install the rest of dependecies
```bash
bash install.sh
```

## Run
To run the server call 
```bash
source .venv/bin/activate
python run.py --config=./config.yaml --port=8080 --host="localhost"
```


## Test
First make sure you have created test-config.yaml inside the tests/ like in the example test-config-example.yaml <br/>
Then call `run_unit_tests` that run each test individual because they will fail if you try to run them by calling just pytest
```bash
bash run_unit_tests.sh
```
Then call `run_integration_tests` only if the server is running and its DB is in memory mode.
```bash
bash run_integration_tests.sh
```

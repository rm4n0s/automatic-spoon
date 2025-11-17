# Automatic Spoon
A simple REST API that will create images, control poses, and remove backgrounds and keep a queue of jobs.

## License
The project has the controversial SSPLv1 license. <br/>
It is like based on GPLv3 license with one caveat on 13th paragraph. If you use the software as a service, then publish all the scripts and source code to replicate it on another hardware. <br/>
For more information read the 'LICENSE' file.

## Installation
With pyenv install python version 3.12.12 and then run these command lines to install dependencies
Install first torch for your GPU or run this script but it may not install the correct torch version
```bash
bash install-torch.sh
```
Run this command line and if it says True in the end it means that GPU will be utilized successfully 
```bash 
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available() if torch.cuda.is_available() else "CPU only")'
```
Now install the rest of dependecies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
bash install.sh
```

## Run
```bash
source .venv/bin/activate
run main.py
```


## Test
First make sure you have created test-config.yaml inside the tests/ like in the example test-config-example.yaml <br/>
Afte that execute
```bash
pytest -s --log-cli-level=DEBUG
```

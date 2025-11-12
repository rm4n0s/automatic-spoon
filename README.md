# Automatic Spoon
A simple REST API that will create images, control poses, and remove backgrounds and keep a queue of jobs.

## License
The project has the controversial SSPLv1 license. <br/>
It is like based on GPLv3 license with one caveat on 13th paragraph. If you use the software as a service, then publish all the scripts and source code to replicate it on another hardware. <br/>
For more information read the 'LICENSE' file.

## Installation
With pyenv install python version 3.12.12 and then run these command lines to install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
bash install.sh
```
Install torch for your GPU or run this script but it may not install the correct torch version
```bash
bash install-torch.sh
```
## Run
```bash
source .venv/bin/activate
run main.py
```


## Test
First make sure you have created test-config.yaml like in the example test-config-example.yaml <br/>
Afte that execute
```bash
pytest -s --log-cli-level=DEBUG
```

## Problems
When deleting .venv and recreating, some times it throws an error that can't find CUDA driver or something like that.<br/>
So I clean the packages and try again

```bash
rm -rf ~/.cache/pip
pip cache purge
pip uninstall rocm torch torchaudio torchvision
bash install.sh
```

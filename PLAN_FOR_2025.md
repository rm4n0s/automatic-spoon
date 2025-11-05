# Plan
The plan for creating automatic-spoon.

## Current Goals
- Make sure it works on my PC first
- Start with the most usefull functionality

## Architecture
### CLI
##### --config <file>

### Configuration
##### for database
- folder to save sqlite
##### for images
- folder to save created images
- expiration of images
- delete image after N downloads
##### for blobs
- folder to save temporary blobs from openpose, midaspose etc
##### for server
- port for the server

### Database
##### Model

##### Engine

##### Job
    - job_id
    - image_id
    - status

##### Image
    - image_id
    - engine_id
    - prompt
    - negative_prompt
    - loras: []model_id
    - embeddings: []model_id
    - reference_image_path
    - poses
        - openpose_path
        - midaspose_path
        - mediapipe_path
    - remove_background: onnx from hugging face


### REST API
#### POST /v1/models
To add a checkpoint in the database.

Input:<br/>
- name
- path
    - http:// to download and save it
    - file:// for local path
    - if none, then it is for Stable Diffusion to download
- model_type (currently only Checkpoint, LoRA, VAE and Embedding)
- model_base (currently only SD and SDXL)
- variant (fp16 or fp32)
- tags

Output:<br/>
- model_id
- name
- status: Downloading, Ready, Error
- path
- model_type
- model_base
- variant
- tags

Errors:<br/>
- error on empty or wrong type field
- failed to find or download file


#### GET /v1/models
Output:<br/>
- List:
    - model_id
    - name
    - status: Downloading, Ready, Error
    - path
    - error
    - model_type
    - model_base
    - variant

#### POST /v1/engines
To create an engine that will keep a checkpoint in VRAM ready to accept jobs for creating images.

Input:<br/>
- checkpoint: model_id
- vae: model_id
- loras: []model_id
- embeddings: []model_id
- long_prompt_technique: compel | sd_embed
- control_net: openpose | midaspose | mediapipepose
- default scheduler
- default steps
- default cfg
- default width and height


Output:<br/>
- engine_id
- status: Ready| Working |Closed
- checkpoint: model_id
- vae: model_id
- loras: []model_id
- embeddings: []model_id
- long_prompt_technique: compel | sd_embed
- control_net: openpose | midaspose | mediapipepose
- default scheduler
- default steps
- default cfg
- default width and height

Errors:<br/>
- Checkpoint doesn't exist
- Not enough VRAM to start
- base of models are not the same between them

#### GET /v1/engines
Output:<br/>
- List
    - engine_id
    - status: ready, busy,stopped
    - checkpoint: model_id
    - vae: model_id
    - loras: []model_id
    - embeddings: []model_id
    - scheduler
    - default steps
    - default cfg
    - default width and height
    - created_at


#### PATCH /v1/engines/{engine_id}/start

#### PATCH /v1/engines/{engine_id}/stop

#### DELETE /v1/engines/{engine_id}
It will delete the engine and any jobs assigned to it

Errors:<br/>
- It will not delete an engine that is running

#### POST /v1/jobs 
To start a job for image creation <br/>
Input:<br/>
- engine_id
- prompt
- negative_prompt
- loras: []modelID
- embeddings : []modelID
- vae : modelID
- reference_image_blob
- pose_image_blob
- type_pose
- remove_background_models : []modelID
- image_file_type: jpeg | png
    
Output:<br/>
- job_id
- status: Waiting, Processing, Finished
- image_id
- created_at
- started_at
- finished_at

Errors:<br/>
- engine doesn't exist

#### GET /v1/jobs
#### GET /v1/images/{image_id}/download
#### GET /v1/images/{image_id}
Output: <br/>
- engine_id
- prompt
- negative_prompt
- loras
- embeddings
- reference_image_path
- pose_image_path
- type_pose
- remove_background: onnx from hugging face
- image_file_type: jpg | png


### Logic
Internally the server will have 2 main processes, the REST API and the Manager:<br/>
The REST API will do two things: <br/>
    - send commands to Manager to create/start/stop/delete engines, aimodels and jobs
    - read the DB after user's request
The Manager will: <br/>
    - listen for commands and for results from the engines
    - start and stop other processes that will run engines
    - will write and update all the tables

#### On start
- start the Manager of engines
    - update all engines from Ready/Working to Closed
    - make all jobs from Processing to Waiting
- start the REST API
- enable queue between master and rest api

#### On end
- the manager will wait for jobs that processing to end and then close engines



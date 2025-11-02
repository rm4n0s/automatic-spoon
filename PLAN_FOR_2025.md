# Plan
The plan for creating automatic-spoon.

## Current Goals
- Make sure it works on my PC first
- Start with the most usefull functionality

## Architecture

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
- default scheduler
- default steps
- default cfg
- default width and height


Output:<br/>
- engine_id
- status: Ready, Working,Closed
- checkpoint: model_id
- vae: model_id
- loras: []model_id
- embeddings: []model_id
- prompt_weighter: [compel or sd_embed]
- controlnet: []TypeOfControlNet (like openpose, mediapipe, midaspose)
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
    - status: Ready, Working,Closed
    - checkpoint: model_id
    - vae: model_id
    - loras: []model_id
    - embeddings: []model_id
    - scheduler
    - default steps
    - default cfg
    - default width and height

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
- loras
- embeddings
- reference_image_blob
- poses
    - openpose_blob
    - midaspose_blob
    - mediapipe_blob
- remove_background: onnx from hugging face
    
Errors:<br/>
- engine doesn't exist

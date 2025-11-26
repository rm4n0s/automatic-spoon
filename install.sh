#!/bin/bash

pip install git+https://github.com/rm4n0s/automatic_spoon_client_sync.git@main

# Install dependencies for development
pip install pytest==8.4.2 \
            pytest-asyncio==1.2.0 \
            httpx==0.28.1

# Install dependencies for CLI
pip install click==8.3.0

# Install dependencies for REST API and UI
pip install nicegui==3.1.0 \
            fastapi==0.120.0 \
            starlette==0.48.0 \
            uvicorn==0.38.0 \
            tortoise-orm==0.25.1 \
            pydantic==2.12.3 \
            mashumaro==3.17 \
            dishka==1.7.2



# Install dependencies for error handling
pip install pytsterrors==0.4.1

# Install dependencies for pose control
echo '+++++++++++++++++ Installing packages to control poses'
pip install controlnet_aux==0.0.10 \
            mediapipe==0.10.21


# Install dependencies for having long prompts
echo '+++++++++++++++++ Installing packages for long prompts'
pip install git+https://github.com/xhinker/sd_embed.git@main \
            compel==2.2.1

# Install dependencies for Stable Diffusion
echo '+++++++++++++++++ Installing stable diffusion'
pip install diffusers==0.35.2 \
    transformers==4.57.1 \
    accelerate==1.11.0

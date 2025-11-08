#!/bin/bash

# Install dependencies for development
pip install pytest==8.4.2 \
            pytest-asyncio==1.2.0

# Install dependencies for CLI
pip install click==8.3.0

# Install dependencies for REST API and UI
pip install nicegui==3.1.0 \
            fastapi==0.120.0 \
            starlette == 0.48.0 \
            uvicorn==0.38.0 \
            tortoise-orm==0.25.1 \
            pydantic==2.12.3 \
            mashumaro==3.17



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

## Install for torch for gfx1151 GPU (for my personal laptop ASUS ROG Flow Z13)
if lspci -nn | grep -q '\[1002:1586\]'; then
    echo '+++++++++++++++++ GFX1151 GPU detected. Installing specific torch dependencies to fix OOMs '
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision
    # pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"
    # pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ --pre torch torchaudio torchvision
else
    # pip install torch torchaudio torchvision
    echo '+++++++++++++++++ INSTALL torch torchaudio torchvision BASED ON YOUR GPU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! '
fi

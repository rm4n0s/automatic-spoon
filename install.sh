#!/bin/bash

# Install dependencies for CLI
pip install click==8.3.0

# Install dependencies for REST API and UI
pip install nicegui==3.1.0 \
            fastapi==0.120.0 \
            uvicorn==0.38.0 \
            tortoise-orm==0.25.1 \
            pydantic==2.12.3 \
            pyyaml==6.0.3



# Install dependencies for error handling
pip install pytsterrors==0.3.0

# Install dependencies for removing background
#
# This is the only order to install rembg with opencv-python
echo 'Installing packages to remove backgrounds'
pip install opencv-python==4.12.0.88
pip install --no-deps rembg==2.0.67
### install rembg dependencies
pip install jsonschema==4.25.1 \
            numpy==2.2.6 \
            pillow==11.3.0 \
            pooch==1.8.2 \
            pymatting==1.1.14 \
            scikit-image==0.25.2 \
            scipy==1.16.3 \
            tqdm==4.67.1 


## Install rembg's onnxruntime dependency           
### Detect NVIDIA driver (requires proprietary driver for CUDA)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU driver detected. Installing onnxruntime-gpu..."
    pip install onnxruntime-gpu

### Detect AMD ROCm driver
elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
    echo "AMD ROCm GPU driver detected. Installing onnxruntime-rocm..."
    pip install onnxruntime-rocm

### Fallback to CPU version if neither is detected
else
    echo "No NVIDIA or AMD ROCm GPU driver detected. Installing CPU version..."
    pip install onnxruntime
fi

# Install dependencies for pose control
echo 'Installing packages to control poses'
pip install controlnet_aux==0.0.10 \
            mediapipe==0.10.21 


# Install dependencies for having long prompts
echo 'Installing packages for long prompts'
pip install git+https://github.com/xhinker/sd_embed.git@main \
            compel==2.2.1

# Install dependencies for Stable Diffusion
echo 'Installing stable diffusion'
pip install diffusers==0.35.2 \
    transformers==4.57.1 \
    accelerate==1.11.0

## Install for torch for gfx1151 GPU (for my personal laptop ASUS ROG Flow Z13)
if lspci -nn | grep -q '\[1002:1586\]'; then 
    echo 'GFX1151 GPU detected. Installing specific torch dependencies to fix OOMs '
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision 
else 
    pip install torch torchaudio torchvision 
fi
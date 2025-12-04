## Install for torch for gfx1151 GPU (for my personal laptop ASUS ROG Flow Z13)
if lspci -nn | grep -q '\[1002:1586\]'; then
    echo '+++++++++++++++++ GFX1151 GPU detected. Installing specific torch dependencies to fix OOMs '
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision
    # pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"
    # pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ --pre torch torchaudio torchvision
else
    # This script detects the GPU type and installs PyTorch, torchvision, and torchaudio accordingly.
    # Assumptions:
    # - For NVIDIA: CUDA must be installed, and nvcc must be in PATH.
    # - For AMD: ROCm must be installed, and rocm-smi must be in PATH.
    # - Falls back to CPU installation if no supported GPU is detected or drivers/toolkits are missing.
    # - Uses pip (assumes Python 3 and pip are installed).
    # - Installs the latest stable versions from PyTorch wheels.
    echo "Detecting GPU and installing PyTorch..."

    # Detect GPU vendor using lspci (requires lspci to be installed, common on Linux)
    if command -v lspci &> /dev/null; then
        gpu_info=$(lspci | grep -i --color 'vga\|3d\|2d' | awk -F: '{print $3}' | tr -d '[]')
    else
        echo "lspci not found. Assuming CPU-only installation."
        gpu_info=""
    fi


    if echo "$gpu_info" | grep -iq "amd\|advanced micro devices"; then
        echo "AMD GPU detected."
        if ! command -v rocm-smi &> /dev/null; then
            echo "Error: ROCm not detected. Install ROCm first."
            exit 1
        fi
        # Extract ROCm version (e.g., 6.2)
        rocm_version=$(rocm-smi --showdriverversion | grep "Kernel Driver Version" | awk '{print $4}' | cut -d '.' -f1-2)
        if [ -z "$rocm_version" ]; then
            echo "Error: Could not detect ROCm version."
            exit 1
        fi
        echo "Detected ROCm version: $rocm_version. Installing PyTorch for rocm$rocm_version."
        pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm${rocm_version}/
    else
        # CPU fallback (includes Intel GPUs or no GPU)
        echo "No supported GPU detected or falling back to CPU installation."
        pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    if [ $? -eq 0 ]; then
        echo "Installation completed successfully."
        echo "Verify with: python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available() if torch.cuda.is_available() else \"CPU only\")'"
    else
        echo "Installation failed. Check errors above."
    fi
fi

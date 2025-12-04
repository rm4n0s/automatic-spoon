
if command -v lspci &> /dev/null; then
    gpu_info=$(lspci | grep -i --color 'vga\|3d\|2d' | awk -F: '{print $3}' | tr -d '[]')
else
    echo "lspci not found. Assuming CPU-only installation."
    gpu_info=""
fi

if echo "$gpu_info" | grep -iq "nvidia"; then
    echo "NVIDIA GPU detected."
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: NVIDIA drivers not detected. Install NVIDIA drivers and CUDA toolkit first."
        exit 1
    fi
    if ! command -v nvcc &> /dev/null; then
        echo "Error: CUDA (nvcc) not detected. Install CUDA toolkit first."
        exit 1
    fi
    # Extract CUDA version (e.g., 12.4)
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d ',' -f1)
    cuda_major_minor=$(echo "$cuda_version" | cut -d '.' -f1-2)
    cuda_compact=$(echo "$cuda_major_minor" | tr -d '.')
    echo "Detected CUDA version: $cuda_version. Installing PyTorch for cu$cuda_compact."
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${cuda_compact}
fi

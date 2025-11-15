from dataclasses import dataclass

import torch


@dataclass
class GPU:
    id: int
    name: str
    total_vram_gb: float


def list_gpus() -> list[GPU]:
    # List available GPUs and their VRAM

    if not torch.cuda.is_available():
        return []
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    gpus: list[GPU] = []
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        # Convert bytes to GB
        total_vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        gpu = GPU(id=i, name=device_name, total_vram_gb=total_vram)
        gpus.append(gpu)

    return gpus

import torch

from .schemas import GPUSchema


class GPUService:
    def list_gpus(self) -> list[GPUSchema]:
        # List available GPUs and their VRAM
        if not torch.cuda.is_available():
            return []
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        gpus: list[GPUSchema] = []
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            total_vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu = GPUSchema(
                id=i,
                name=device_name,
                total_vram_gb=total_vram,
            )
            gpus.append(gpu)

        return gpus

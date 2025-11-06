import multiprocessing
import asyncio
from src import models, config


class Manager:
    def __init__(self, conf: config.Config, view_queue: multiprocessing.Queue[str]):
        self.queue = view_queue
        self.config = config


def start_manager(conf: config.Config, view_queue: multiprocessing.Queue[str]):
    manager = Manager(conf, view_queue)



# def _organize_models(self):
#         for (model, info) in self._aimodels:
#             match model.model_type:
#                 case models.AIModelType.CHECKPOINT:
#                     if self._checkpoint is None: 
#                         self._checkpoint = model
#                     else:
#                         raise TSTError("checkpoint-redeclared", "Checkpoint in generator has been redeclared on the start")
                    
#                 case models.AIModelType.VAE:
#                     if self._vae is None: 
#                         self._vae = model
#                     else:
#                         raise TSTError("vae-redeclared", "VAE in generator has been redeclared on the start")
                    
#                 case models.AIModelType.CONTROLNET_MIDAS:
#                     self._cnets.append(model)
#                 case models.AIModelType.CONTROLNET_OPENPOSE:
#                     self._cnets.append(model)
#                 case models.AIModelType.EMBEDDING:
#                     self._embeddings.append(model)
#                 case models.AIModelType.LORA:
#                     lg = LoraForGenerator(model=model, weight=info.weight)
#                     self._loras.append(lg)
#                 case models.AIModelType.REMBG:
#                     if self._rembg is None: 
#                         self._rembg = model
#                     else:
#                         raise TSTError("rembg-model-redeclared", "Rembg model in generator has been redeclared on the start")
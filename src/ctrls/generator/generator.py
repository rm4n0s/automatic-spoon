import multiprocessing
from .commands import EngineCommands, JobCommand
from .pipe import create_controlnets, create_vae, create_pipe, set_scheduler
from src import models
from pytsterrors import TSTError
from diffusers import DiffusionPipeline

class Generator:
    _command_queue: multiprocessing.Queue[tuple[EngineCommands, JobCommand]]
    _engine: models.Engine
    _aimodels: list[models.AIModel]
    _checkpoint:models.AIModel | None = None 
    _vae :models.AIModel | None = None
    _cnets :list[models.AIModel]  = []
    _embeddings: list[models.AIModel] = []
    _loras: list[models.AIModel] = []
    _rembg :models.AIModel | None = None
    def __init__(self, engine: models.Engine, aimodels: list[models.AIModel], commands_queue: multiprocessing.Queue[tuple[EngineCommands, JobCommand]]):
        self._command_queue = commands_queue
        self._engine = engine
        self._aimodels = aimodels


    def _organize_models(self):
        for v in self._aimodels:
            match v.model_type:
                case models.AIModelType.CHECKPOINT:
                    if self._checkpoint is None: 
                        self._checkpoint = v
                    else:
                        raise TSTError("checkpoint-redeclared", "Checkpoint in generator has been redeclared on the start")
                    
                case models.AIModelType.VAE:
                    if self._vae is None: 
                        self._vae = v
                    else:
                        raise TSTError("vae-redeclared", "VAE in generator has been redeclared on the start")
                    
                case models.AIModelType.CONTROLNET_MIDAS:
                    self._cnets.append(v)
                case models.AIModelType.CONTROLNET_OPENPOSE:
                    self._cnets.append(v)
                case models.AIModelType.EMBEDDING:
                    self._embeddings.append(v)
                case models.AIModelType.LORA:
                    self._loras.append(v)
                case models.AIModelType.REMBG:
                    if self._rembg is None: 
                        self._rembg = v
                    else:
                        raise TSTError("rembg-model-redeclared", "Rembg model in generator has been redeclared on the start")
                    
    def _create_pipe(self) -> DiffusionPipeline:
        if self._checkpoint is None:
            raise TSTError("checkpoint-is-none", "There is no checkpoint for the generator to start the pipe")
        
        vae = None
        if self._vae is not None:
            vae = create_vae(self._vae)

        controlnets =[]
        if len(self._cnets) > 0:
            controlnets = create_controlnets(self._cnets)

        
        pipe = create_pipe(self._checkpoint, vae, controlnets)
        set_scheduler(pipe, self._engine.scheduler)
        pipe.to("cuda")
        pipe.safety_checker = None
        return pipe

                        
    def start(self):
        self._organize_models()
        self._create_pipe()





def start_generator(engine: models.Engine, aimodels: list[models.AIModel], commands_queue: multiprocessing.Queue[tuple[EngineCommands, JobCommand]]):
    generator = Generator(engine, aimodels, commands_queue)

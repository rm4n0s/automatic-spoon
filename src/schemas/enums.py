import enum


class AIModelStatus(enum.StrEnum):
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class Variant(enum.StrEnum):
    FP16 = "fp16"
    FP32 = "fp32"


class PathType(enum.StrEnum):
    HUGGING_FACE = "hugging_face"
    FILE = "file"


class AIModelType(enum.StrEnum):
    CHECKPOINT = "checkpoint"
    EMBEDDING = "embedding"
    VAE = "vae"
    LORA = "lora"
    CONTROLNET = "controlnet"


class AIModelBase(enum.StrEnum):
    ALL = "all"
    SD = "sd"
    SDXL = "sdxl"


class LongPromptTechnique(enum.StrEnum):
    COMPEL = "compel"
    SDEMBED = "sdembed"


class ControlNetType(enum.StrEnum):
    MIDAS = "midas"
    OPENPOSE = "openpose"
    MEDIAPIPE = "mediapipe"


class Scheduler(enum.StrEnum):
    EULERA = "eulera"
    EULER = "euler"
    LMS = "lms"
    HEUN = "heun"
    DPM2 = "dpm2"
    DPM2A = "dpm2a"
    DPM2SA = "dpm2sa"
    DPM2M = "dpm2m"
    DPMSDE = "dpmsde"
    # DPMFAST = "dpmfast" # from k_diffusion
    # DPMADAPTIVE = "dpmadaptive" # from k_diffusion
    LMSKARRAS = "lmskarras"
    DPM2KARRAS = "dpm2karras"
    DPM2AKARRAS = "dpm2akarras"
    DPM2SAKARRAS = "dpm2sakarras"
    DPM2MKARRAS = "dpm2mkarras"
    DPMSDEKARRAS = "dpmsdekarras"
    DDIM = "ddim"
    PLMS = "plms"
    UNIPC = "unipc"
    LCM = "lcm"
    DDPM = "ddpm"
    DEIS = "deis"


class JobStatus(enum.StrEnum):
    WAITING = "waiting"
    PROCESSING = "processing"
    FINISHED = "finished"


class FileImageType(enum.StrEnum):
    JPG = "jpg"
    PNG = "png"


class EngineCommandType(enum.StrEnum):
    JOB = "job"
    CLOSE = "close"


class EngineResultType(enum.StrEnum):
    JOB = "job"
    ERROR = "error"
    CLOSED = "closed"

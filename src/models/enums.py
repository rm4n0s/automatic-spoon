import enum

class AIModelStatus(enum.StrEnum):
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"

class Variant(enum.StrEnum):
    FP8 = "fp8"
    FP16= "fp16"
    FP32 = "fp32"

class AIModelType(enum.StrEnum):
    CHECKPOINT = "checkpoint"
    EMBEDDING  = "embedding"
    VAE = "vae"
    LORA = "lora"
    CONTROLNET_OPENPOSE = "controlnet_openpose"
    CONTROLNET_MIDAS = "controlnet_midas"
    REMBG = "rembg"

class AIModelBase(enum.StrEnum):
    SD = "sd"
    SDXL = "sdxl"

class LongPromptTechnique(enum.StrEnum):
    NONE = "none"
    COMPEL = "compel"
    SDEMBED = "sdembed"

class ControlNetPose(enum.StrEnum):
    NONE = "none"
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
    DPMFAST = "dpmfast" # from k_diffusion
    DPMADAPTIVE = "dpmadaptive" # from k_diffusion
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


class EngineStatus(enum.StrEnum):
    Ready = "ready"
    Working = "working"
    Closed = "closed"

class JobStatus(enum.StrEnum):
    WAITING = "waiting"
    PROCESSING = "processing"
    FINISHED = "finished"

class FileImageType(enum.StrEnum):
    JPG = "jpg"
    PNG = "png"
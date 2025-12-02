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
    CANNY = "canny"


class PipeType(enum.StrEnum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"


class Scheduler(enum.StrEnum):
    EULERA = "eulera"
    EULER = "euler"
    LMS = "lms"
    HEUN = "heun"
    DPM2 = "dpm2"
    DPM2A = "dpm2a"
    DPM2SA = "dpm2sa"
    DPM2M = "dpm2m"
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


class GeneratorStatus(enum.StrEnum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    CLOSING = "closing"
    CLOSED = "closed"


class FileImageType(enum.StrEnum):
    JPG = "jpg"
    PNG = "png"


class GeneratorCommandType(enum.StrEnum):
    JOB = "job"
    CLOSE = "close"


class GeneratorResultType(enum.StrEnum):
    READY = "ready"
    JOB_STARTING = "job_starting"
    JOB_FINISHED = "job_finished"
    IMAGE_FINISHED = "image_finished"
    ERROR = "error"
    CRASH = "crash"
    CLOSED = "closed"


class ManagerSignalType(enum.StrEnum):
    NEW_JOB = "new_job"
    CHECK_WAITING_JOBS = "check_waiting_jobs"

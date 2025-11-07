from dataclasses import dataclass

from mashumaro.mixins.yaml import DataClassYAMLMixin


@dataclass
class ModelPath:
    file_path: str | None = None
    hugging_face: str | None = None
    weight: float | None = None
    trigger_pos: str | None = None
    trigger_neg: str | None = None


@dataclass
class TestConfig(DataClassYAMLMixin):
    vae_sd: ModelPath
    vae_sdxl: ModelPath
    checkpoint_sd: ModelPath
    checkpoint_sdxl: ModelPath
    openpose: ModelPath
    mediapipe: ModelPath
    midas: ModelPath
    loras_sd: list[ModelPath]
    loras_sdxl: list[ModelPath]
    embeddings_sd: list[ModelPath]
    embeddings_sdxl: list[ModelPath]


def read_test_config(filepath: str) -> TestConfig:
    with open(filepath, "r") as file:
        yaml_content = file.read()
        config = TestConfig.from_yaml(yaml_content)
        return config

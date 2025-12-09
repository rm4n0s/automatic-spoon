# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from mashumaro.mixins.yaml import DataClassYAMLMixin


@dataclass
class ModelPath:
    file_path: str | None = None
    hugging_face: str | None = None
    weight: float | None = None
    trigger_pos: str | None = None
    trigger_neg: str | None = None
    low_threshold: int | None = None
    high_threshold: int | None = None


@dataclass
class TestConfig(DataClassYAMLMixin):
    gpu_id: int
    vae_sd: ModelPath
    vae_sdxl: ModelPath
    checkpoint_sd: ModelPath
    checkpoint_sdxl: ModelPath
    checkpoint_v_pred_sdxl: ModelPath
    openpose_sd: ModelPath
    openpose_sdxl: ModelPath
    mediapipe_sd: ModelPath
    mediapipe_sdxl: ModelPath
    midas_sd: ModelPath
    midas_sdxl: ModelPath
    canny_sd: ModelPath
    canny_sdxl: ModelPath
    loras_sd: list[ModelPath]
    loras_sdxl: list[ModelPath]
    embeddings_sd: list[ModelPath]
    embeddings_sdxl: list[ModelPath]


def read_test_config(filepath: str) -> TestConfig:
    with open(filepath, "r") as file:
        yaml_content = file.read()
        config = TestConfig.from_yaml(yaml_content)
        return config

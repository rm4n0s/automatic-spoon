# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

import os
from dataclasses import dataclass

from mashumaro.mixins.yaml import DataClassYAMLMixin


@dataclass
class Config(DataClassYAMLMixin):
    db_path: str
    images_path: str
    poses_path: str
    hugging_face_path: str


def read_config(filepath: str) -> Config:
    with open(filepath, "r") as file:
        yaml_content = file.read()
        config = Config.from_yaml(yaml_content)
        return config


def enable_hugging_face_envs(cfg: Config):
    os.environ["HF_HOME"] = cfg.hugging_face_path
    # os.environ["HUGGINGFACE_HUB_CACHE"] = cfg.hugging_face_path + "/hub"
    # os.environ["TRANSFORMERS_CACHE"] = cfg.hugging_face_path + "/transformers"
    # os.environ["DIFFUSERS_CACHE"] = cfg.hugging_face_path + "/diffusers"

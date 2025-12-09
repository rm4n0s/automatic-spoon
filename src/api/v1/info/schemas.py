# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from pydantic import BaseModel


class InfoSchema(BaseModel):
    db_path: str
    images_path: str
    hugging_face_path: str

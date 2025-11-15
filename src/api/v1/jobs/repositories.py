import base64
import os
import uuid

from pydantic_core.core_schema import GeneralPlainInfoSerializerFunction

from src.api.v1.aimodels.schemas import AIModelSchema
from src.core.config import Config
from src.db.models import AIModel, ControlNetImage, Image, Job

from .schemas import ControlNetImageSchema, ImageSchema, JobSchema
from .user_inputs import JobUserInput


class JobRepo:
    async def get_all(self) -> list[JobSchema]:
        return []

    async def create(self, config: Config, input: JobUserInput) -> JobSchema:
        job_db = await Job.create(generator_id=input.generator_id)
        list_img_sch = []
        for img_input in input.images:
            img_filename = str(uuid.uuid4())
            kwargs = {}
            kwargs["generator_id"] = (input.generator_id,)
            kwargs["job_id"] = (job_db.id,)
            kwargs["file_path"] = (
                os.path.join(
                    config.images_path, img_filename + "." + img_input.file_type
                ),
            )
            kwargs["prompt"] = img_input.prompt
            kwargs["negative_prompt"] = img_input.negative_prompt
            if img_input.seed:
                kwargs["seed"] = img_input.seed

            if img_input.guidance_scale:
                kwargs["guidance_scale"] = img_input.guidance_scale

            if img_input.width:
                kwargs["width"] = img_input.width

            if img_input.height:
                kwargs["height"] = img_input.height

            if img_input.steps:
                kwargs["steps"] = img_input.steps

            if img_input.control_guidance_start:
                kwargs["control_guidance_start"] = img_input.control_guidance_start

            if img_input.control_guidance_end:
                kwargs["control_guidance_end"] = img_input.control_guidance_end

            img_db = await Image.create(**kwargs)

            ci_dbs = []
            for ci in img_input.control_images:
                pose_filename = "pose-" + str(uuid.uuid4())
                pose_binary = base64.b64decode(ci.data_base64)
                pose_file_path = os.path.join(config.images_path, pose_filename)
                with open(pose_file_path, "wb") as output_file:
                    _ = output_file.write(pose_binary)

                ci_db = await ControlNetImage.create(
                    image_id=img_db.id,
                    aimodel_id=ci.aimodel_id,
                    controlnet_conditioning_scale=ci.controlnet_conditioning_scale,
                    file_path=pose_file_path,
                )
                ci_dbs.append(ci_db)

            img_sch = await serialize_image(img_db, ci_dbs)

            list_img_sch.append(img_sch)

        return JobSchema(
            id=job_db.id, generator_id=job_db.generator_id, images=list_img_sch
        )


async def serialize_image(img_db: Image, cni_dbs: list[ControlNetImage]) -> ImageSchema:
    img_sch = ImageSchema(
        id=img_db.id,
        job_id=img_db.job_id,
        generator_id=img_db.generator_id,
        prompt=img_db.prompt,
        negative_prompt=img_db.negative_prompt,
        ready=img_db.ready,
        file_path=img_db.file_path,
        seed=img_db.seed,
        guidance_scale=img_db.guidance_scale,
        width=img_db.width,
        height=img_db.height,
        steps=img_db.steps,
        file_type=img_db.file_type,
        control_guidance_start=img_db.control_guidance_start,
        control_guidance_end=img_db.control_guidance_end,
    )

    list_cni_sch = []
    for cni_db in cni_dbs:
        aimodel_sch = None
        if cni_db.aimodel_id:
            aimodel = await AIModel.get_or_none(id=cni_db.aimodel_id)
            if aimodel:
                aimodel_sch = AIModelSchema.model_validate(aimodel)

        cni_sch = ControlNetImageSchema(
            aimodel=aimodel_sch,
            image_file_path=cni_db.file_path,
            controlnet_conditioning_scale=cni_db.controlnet_conditioning_scale,
        )
        list_cni_sch.append(cni_sch)

    img_sch.control_images = list_cni_sch
    return img_sch

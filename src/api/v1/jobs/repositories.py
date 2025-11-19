import base64
import os
import uuid
from typing import Any

from pytsterrors import TSTError
from tortoise.expressions import Q

from src.api.v1.aimodels.schemas import AIModelSchema
from src.db.models import AIModel, ControlNetImage, Image, Job

from .schemas import ControlNetImageSchema, ImageSchema, JobSchema
from .user_inputs import JobUserInput


class JobRepo:
    async def get_all(self) -> list[JobSchema]:
        jobs = await Job.all()
        job_sch_list = []
        for job in jobs:
            imgs = await Image.filter(job_id=job.id)
            img_sch_list = []
            for img in imgs:
                cnis = await ControlNetImage.filter(job_id=job.id, image_id=img.id)
                img_sch = await serialize_image(img, cnis)
                img_sch_list.append(img_sch)

            job_sch = JobSchema(
                id=job.id,
                generator_id=job.generator_id,
                images=img_sch_list,
                status=job.status,
            )
            job_sch_list.append(job_sch)
        return job_sch_list

    async def filter(self, *args: Q, **kwargs: Any) -> list[JobSchema]:  # pyright: ignore[reportExplicitAny]
        jobs = await Job.filter(*args, **kwargs)
        job_sch_list = []
        for job in jobs:
            imgs = await Image.filter(job_id=job.id)
            img_sch_list = []
            for img in imgs:
                cnis = await ControlNetImage.filter(job_id=job.id, image_id=img.id)
                img_sch = await serialize_image(img, cnis)
                img_sch_list.append(img_sch)

            job_sch = JobSchema(
                id=job.id,
                generator_id=job.generator_id,
                images=img_sch_list,
                status=job.status,
            )
            job_sch_list.append(job_sch)
        return job_sch_list

    async def get_or_none(self, id: int) -> JobSchema | None:
        job = await Job.get_or_none(id=id)
        if job is None:
            return None

        imgs = await Image.filter(job_id=job.id)
        img_sch_list = []
        for img in imgs:
            cnis = await ControlNetImage.filter(image_id=img.id)
            img_sch = await serialize_image(img, cnis)
            img_sch_list.append(img_sch)

        job_sch = JobSchema(
            id=job.id,
            generator_id=job.generator_id,
            images=img_sch_list,
            status=job.status,
        )
        return job_sch

    async def create(self, images_folder_path: str, input: JobUserInput) -> JobSchema:
        job_db = await Job.create(generator_id=input.generator_id)
        img_sch_list = []
        for img_input in input.images:
            img_filename = str(uuid.uuid4())
            kwargs = {}
            kwargs["generator_id"] = input.generator_id
            kwargs["job_id"] = job_db.id
            kwargs["file_path"] = os.path.join(
                images_folder_path, img_filename + "." + img_input.file_type
            )
            kwargs["prompt"] = img_input.prompt
            kwargs["negative_prompt"] = img_input.negative_prompt
            kwargs["file_type"] = img_input.file_type
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
                pose_file_path = os.path.join(images_folder_path, pose_filename)
                with open(pose_file_path, "wb") as output_file:
                    _ = output_file.write(pose_binary)

                ci_db = await ControlNetImage.create(
                    image_id=img_db.id,
                    job_id=job_db.id,
                    aimodel_id=ci.aimodel_id,
                    controlnet_conditioning_scale=ci.controlnet_conditioning_scale,
                    file_path=pose_file_path,
                )
                ci_dbs.append(ci_db)

            img_sch = await serialize_image(img_db, ci_dbs)

            img_sch_list.append(img_sch)

        return JobSchema(
            id=job_db.id,
            generator_id=job_db.generator_id,
            images=img_sch_list,
            status=job_db.status,
        )

    async def delete(self, job_id: int):
        job = await Job.get_or_none(id=job_id)
        if not job:
            raise TSTError(
                "job-not-found",
                f"Job with ID {job_id} not found",
                metadata={"status_code": 404},
            )

        await job.delete()

        imgs = await Image.filter(job_id=job_id)
        cnets = await ControlNetImage.filter(job_id=job_id)
        for cnet in cnets:
            await cnet.delete()

        for img in imgs:
            await img.delete()

    async def delete_by_generator(self, generator_id: int):
        jobs = await Job.filter(generator_id=generator_id)
        for job in jobs:
            await job.delete()

            imgs = await Image.filter(job_id=job.id)
            cnets = await ControlNetImage.filter(job_id=job.id)
            for cnet in cnets:
                await cnet.delete()

            for img in imgs:
                await img.delete()


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

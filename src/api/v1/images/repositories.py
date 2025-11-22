from pytsterrors import TSTError

from src.api.v1.aimodels.schemas import AIModelSchema
from src.db.models import AIModel, ControlNetImage, Image

from .schemas import ControlNetImageSchema, ImageSchema


class ImageRepo:
    async def get_all(self) -> list[ImageSchema]:
        imgs = await Image.all()

        list_image_schemas = []
        for img in imgs:
            cnis = await ControlNetImage.filter(image_id=img.id)
            img_sch = await serialize_image(img, cnis)
            list_image_schemas.append(img_sch)
        return list_image_schemas

    async def get_one(self, id: int) -> ImageSchema:
        img = await Image.get_or_none(id=id)
        if img is None:
            raise TSTError(
                "image-not-found",
                f"Image with ID {id} not found",
                metadata={"status_code": 404},
            )

        cnis = await ControlNetImage.filter(image_id=img.id)
        return await serialize_image(img, cnis)


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

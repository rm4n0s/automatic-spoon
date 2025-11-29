from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status
from fastapi.responses import FileResponse

from src.core.enums import FileImageType

from .repositories import ImageRepo
from .schemas import ImageSchema

router = APIRouter()


@router.get("", response_model=list[ImageSchema])
@inject
async def get_images(repo: FromDishka[ImageRepo]):
    return await repo.get_all()


@router.get("/{id}", response_model=ImageSchema)
@inject
async def get_one_image(id: int, repo: FromDishka[ImageRepo]):
    return await repo.get_one(id)


@router.get("/{id}/show")
@inject
async def show_image(id: int, repo: FromDishka[ImageRepo]):
    img = await repo.get_one(id)
    media_type = ""
    match img.file_type:
        case FileImageType.PNG:
            media_type = "images/png"
        case FileImageType.JPG:
            media_type = "images/jpeg"

    return FileResponse(img.file_path, media_type=media_type)

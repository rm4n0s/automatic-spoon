from automatic_spoon_client_sync import (
    AIModelBase,
    AIModelCaller,
    AIModelType,
    AIModelUserInput,
    CreationError,
    PathType,
    Variant,
)


def test_aimodel_validate_empty_fields():
    host = "http://localhost:8080"
    aimodel_caller = AIModelCaller(host)

    threw_exception = False
    try:
        input = AIModelUserInput(
            name="",
            path="",
            path_type=PathType.FILE,
            variant=Variant.FP16,
            model_base=AIModelBase.SDXL,
            model_type=AIModelType.CHECKPOINT,
        )

        aimodel_caller.create_aimodel(input)
    except CreationError as exc:
        threw_exception = True
        errs = {}
        for errf in exc.error_fields:
            errs[errf.field] = errf.error

        assert "name" in errs.keys() and errs["name"] == "name can't be empty"
        assert "path" in errs.keys() and errs["path"] == "path can't be empty"

        assert len(errs) == 2

    assert threw_exception


def test_aimodel_validate_non_existing_file():
    host = "http://localhost:8080"
    aimodel_caller = AIModelCaller(host)
    input = AIModelUserInput(
        name="hello",
        path="/non/existing/file",
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_base=AIModelBase.SDXL,
        model_type=AIModelType.CHECKPOINT,
    )

    threw_exception = False
    try:
        aimodel_caller.create_aimodel(input)
    except CreationError as exc:
        threw_exception = True
        errs = {}
        for errf in exc.error_fields:
            errs[errf.field] = errf.error

        assert errs["path"] == f"the path {input.path} doesn't exist"
        assert len(errs) == 1

    assert threw_exception


def test_aimodel_validate_already_created_aimodel():
    host = "http://localhost:8080"
    aimodel_caller = AIModelCaller(host)
    input = AIModelUserInput(
        name="hello",
        path="stabilityai/stable-diffusion-xl-base-1.0",
        path_type=PathType.HUGGING_FACE,
        variant=Variant.FP16,
        model_base=AIModelBase.SDXL,
        model_type=AIModelType.CHECKPOINT,
    )

    threw_exception = False
    try:
        aimodel_caller.create_aimodel(input)
    except CreationError as exc:
        assert True

    input = AIModelUserInput(
        name="hello",
        path="stabilityai/stable-diffusion-xl-base-1.0",
        path_type=PathType.HUGGING_FACE,
        variant=Variant.FP16,
        model_base=AIModelBase.SDXL,
        model_type=AIModelType.CHECKPOINT,
    )

    threw_exception = False
    try:
        aimodel_caller.create_aimodel(input)
    except CreationError as exc:
        threw_exception = True
        errs = {}
        for errf in exc.error_fields:
            errs[errf.field] = errf.error

        assert (
            errs["path"] == f"the path {input.path} is already used by another AIModel"
        )
        assert len(errs) == 1

    assert threw_exception

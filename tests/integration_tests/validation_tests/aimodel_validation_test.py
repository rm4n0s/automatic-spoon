from automatic_spoon_client_sync import (
    AIModelBase,
    AIModelCaller,
    AIModelType,
    AIModelUserInput,
    PathType,
    Variant,
)


def test_aimodel_validation_failures():
    host = "http://localhost:8080"
    aimodel_caller = AIModelCaller(host)

    input = AIModelUserInput(
        name="",
        path="",
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_base=AIModelBase.SDXL,
        model_type=AIModelType.CHECKPOINT,
    )

    aimodel_caller.create_aimodel(input)

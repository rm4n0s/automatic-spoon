from automatic_spoon_client_sync.info_caller import InfoCaller

from src.api.v1.aimodels.user_inputs import AIModelUserInput
from src.utils import read_test_config


def test_sdxl_image_creation():
    host = "http://localhost:8080"
    info_caller = InfoCaller(host)

    info = info_caller.get_info()
    assert info.db_path == ":memory:"

    cfg = read_test_config("tests/test-config.yaml")

    assert cfg.checkpoint_sdxl.file_path is not None
    assert cfg.vae_sdxl.file_path is not None

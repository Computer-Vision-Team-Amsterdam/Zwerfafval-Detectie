import logging
import os

from aml_interface.aml_interface import AMLInterface

from zwerfafval_detectie.settings import ZwerfafvalDetectieSettings

logger = logging.getLogger("inference_pipeline")

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)
try:
    ZwerfafvalDetectieSettings.set_from_yaml(config_path)
    settings = ZwerfafvalDetectieSettings.get_settings()
except FileNotFoundError:
    logger.warning(
        "Config file for YoloModelDevelopmentKit not found. If the project was extended this warning can be ignored."
    )

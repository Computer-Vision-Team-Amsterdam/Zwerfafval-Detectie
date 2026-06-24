from typing import Any, Dict, List, Union

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AMLExperimentDetailsSpec(SettingsSpecModel):
    experiment_name: str
    compute_name: str = None
    env_name: str = None
    env_version: int = None
    src_dir: str = None
    ai_instrumentation_key: str = None


class LoggingSpec(SettingsSpecModel):
    loglevel_own: str = "INFO"
    own_packages: List[str] = [
        "__main__",
    ]
    extra_loglevels: Dict[str, str] = {}
    basic_config: Dict[str, Any] = {
        "level": "WARNING",
        "format": "%(asctime)s|%(levelname)-8s|%(name)s|%(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    ai_instrumentation_key: str = ""


class TrainingModelParameters(SettingsSpecModel):
    img_size: int = 960
    batch: Union[float, int] = -1
    epochs: int = 100
    n_classes: int = 3
    name_classes: List[str] = ["zwerfafval_fijn", "zwerfafval_grof", "grofvuil"]
    cache: Union[bool, str] = False
    patience: int = 25
    cos_lr: bool = False
    seed: int = 0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5


class TrainingPipelineSpec(SettingsSpecModel):
    model_parameters: TrainingModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None


class ZwerfafvalDetectieSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    logging: LoggingSpec = LoggingSpec()
    training_pipeline: TrainingPipelineSpec = None

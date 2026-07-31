from typing import Dict, List, Union

from yolo_model_development_kit.settings.settings_schema import (
    AMLExperimentDetailsSpec,
    InferencePipelineSpec,
    LoggingSpec,
    SettingsSpecModel,
)


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
    logging: LoggingSpec = LoggingSpec()
    aml_experiment_details: AMLExperimentDetailsSpec
    inference_pipeline: InferencePipelineSpec
    training_pipeline: TrainingPipelineSpec = None

from cvtoolkit.settings.settings_helper import GenericSettings, Settings
from pydantic import BaseModel

from zwerfafval_detectie.settings.settings_schema import ZwerfafvalDetectieSettingsSpec


class ZwerfafvalDetectieSettings(Settings):  # type: ignore
    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = ZwerfafvalDetectieSettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)

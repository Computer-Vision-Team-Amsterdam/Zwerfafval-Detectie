# settings/__init__.py

from .settings import ZwerfafvalDetectieSettings  # Re-export main settings class
from .settings_schema import (  # Re-export schema classes
    AMLExperimentDetailsSpec,
    LoggingSpec,
    ZwerfafvalDetectieSettingsSpec,
)

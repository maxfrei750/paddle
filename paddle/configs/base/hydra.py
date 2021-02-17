from dataclasses import dataclass

from hydra.conf import HydraConf, RunDir

# TODO: Disable stdout log file.


@dataclass
class HydraRunConfigBase(RunDir):
    dir: str = "logs/${hydra:job.config_name}/${now:%Y-%m-%d_%H-%M-%S}"


@dataclass
class HydraConfigBase(HydraConf):
    run: HydraRunConfigBase = HydraRunConfigBase()

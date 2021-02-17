from dataclasses import dataclass


@dataclass
class ProgramConfigBase:
    random_seed: int = 42
    search_optimum_learning_rate: bool = False

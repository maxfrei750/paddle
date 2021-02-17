from dataclasses import dataclass


@dataclass
class CallbackConfigBase:
    early_stopping_patience: int = 20
    # TODO: split into a section for each callback
    # TODO: Add all parameters of the callbacks

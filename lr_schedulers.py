from torch import optim


class LearningRateWarmup(optim.lr_scheduler.LambdaLR):
    """Larning rate scheduler, which increases the learning rate gradually.

    :param optimizer: optimizer, whose learning rate is being controlled
    :param warmup_duration: number of events until the final learning rate is reached
    :param warmup_factor: determines the starting learning rate:
     start_learning_rate = end_learning_rate * warmup_factor
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_duration: int = 1000,
        warmup_factor: float = 0.001,
    ) -> None:

        self.warmup_duration = warmup_duration
        self.warmup_factor = warmup_factor

        def f(x):
            if x >= self.warmup_duration:
                return 1
            alpha = float(x) / self.warmup_duration
            return self.warmup_factor * (1 - alpha) + alpha

        super(LearningRateWarmup, self).__init__(optimizer, f)

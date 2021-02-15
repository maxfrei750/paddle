import fire

from paddle.inspection import inspect_model

if __name__ == "__main__":
    fire.Fire(inspect_model)

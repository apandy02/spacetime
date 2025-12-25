import os


def maybe_set_wandb_sandbox_key() -> None:
    if "WANDB_API_KEY_SANDBOX" in os.environ:
        print("Using sandbox key")
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY_SANDBOX"]

import os


def maybe_set_wandb_sandbox_key() -> None:
    if "WANDB_API_KEY_SANDBOX" in os.environ:
        print("Using sandbox key")
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY_SANDBOX"]


def is_rank_zero() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


def maybe_disable_wandb_for_non_zero_ranks() -> None:
    if not is_rank_zero():
        os.environ["WANDB_MODE"] = "disabled"

__all__ = ["ACTAdapter", "ACTTrainer"]


def __getattr__(name: str):
    if name == "ACTAdapter":
        from testbed.policies.act.adapter import ACTAdapter

        return ACTAdapter
    if name == "ACTTrainer":
        from testbed.policies.act.trainer import ACTTrainer

        return ACTTrainer
    raise AttributeError(name)

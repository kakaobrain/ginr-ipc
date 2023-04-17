import torch

def create_inr_optimizer(model, config):
    optimizer_type = config.type.lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"{optimizer_type} invalid..")
    return optimizer


def create_optimizer(model, config):
    arch_type = config.arch.type.lower()
    if "inr" in config.arch.type:
        optimizer = create_inr_optimizer(model, config.optimizer)
    else:
        raise ValueError(f"{arch_type} invalid..")
    return optimizer

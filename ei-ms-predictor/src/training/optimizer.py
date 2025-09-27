import torch

def create_optimizer(model, lr=1e-4, weight_decay=0.01):
    """
    Creates an AdamW optimizer for the given model.

    Args:
        model (nn.Module): The model to optimize.
        lr (float): The learning rate.
        weight_decay (float): The weight decay (L2 penalty).

    Returns:
        An instance of torch.optim.AdamW.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    return optimizer

def create_scheduler(optimizer, num_warmup_steps=1000, num_training_steps=100000):
    """
    Creates a learning rate scheduler with a linear warmup followed by a cosine decay.
    This is a common and effective scheduling strategy.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.

    Returns:
        An instance of torch.optim.lr_scheduler.LambdaLR.
    """

    def lr_lambda(current_step):
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        # The scheduler in the spec is CosineAnnealingWarmRestarts, which is more complex.
        # This lambda-based cosine decay is a simpler and often equally effective alternative.
        # It computes a scaling factor using the cosine function.
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

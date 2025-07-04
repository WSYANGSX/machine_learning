from typing import Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LRWarmDampingScheduler(_LRScheduler):
    """Custom learning rate scheduler, including preheating phase and step decay

    Args:
        _LRScheduler (_type_): The base class of all learning rate scheduler class.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        burn_in: int,
        lr_steps: Sequence[int],
        decay_scales: Sequence[int],
        batch_sizes: int,
    ):
        """Initialize LRWarmDampingScheduler class

        Args:
            optimizer (Optimizer): Optimizer to adjust learning rate.
            initial_lr (float): Initial learning rate.
            burn_in (int): The number of batches in the preheating stage.
            lr_steps (Sequence[int]): The number of batches for adjusting the learning rate.
            decay_scales (Sequence[int]): The scaling factor of learning rate adjusts in lr_steps.
        """
        self.initial_lr = initial_lr
        self.burn_in = burn_in
        self.lr_steps = lr_steps
        self.decay_scales = decay_scales
        self.batch_sizes = batch_sizes

        super().__init__(optimizer)

    def step(self, epoch=0):
        """
        Update the learning rate (called after each epoch)
        """
        batches = epoch * self.batch_sizes

        if batches < self.burn_in:
            lr = self.initial_lr * (batches / self.burn_in)
        else:
            lr = self.initial_lr
            for step, scale_factor in zip(self.lr_steps, self.decay_scales):
                if batches > step:
                    lr *= scale_factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        state = super().state_dict()
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

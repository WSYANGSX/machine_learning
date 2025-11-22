import torch
from torch.utils.data import Dataset


class InfiniteDataLoader:
    """Wrap a regular DataLoader into an iterator with an "infinite loop."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = True, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

        self.iterator = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)  # Try to take a batch from the current iterator
        except StopIteration:
            # One round is used up. Build a new iterator, which is equivalent to a "new epoch"
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch

    def __len__(self):
        return len(self.loader)

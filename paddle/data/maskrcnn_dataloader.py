import multiprocessing
from typing import List, Optional, Sequence, Tuple

import torch.utils
from torch.utils.data import Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t

from paddle.custom_types import Annotation, Batch, Image


class MaskRCNNDataLoader(torch.utils.data.DataLoader):
    """Pytorch DataLoader with sensible defaults for MaskRCNN datasets."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: Optional[int] = None,
        collate_fn: _collate_fn_t = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        if collate_fn is None:
            collate_fn = MaskRCNNDataLoader.collate

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    @staticmethod
    def collate(uncollated_batch: List[Tuple[Image, Annotation]]) -> Batch:
        """Defines how to collate batches.

        :param uncollated_batch: Uncollated batch of data. List containing tuples of images and
            annotations.
        :return: Collated batch. Tuple containing a tuple of images and a tuple of annotations.
        """
        return tuple(zip(*uncollated_batch))

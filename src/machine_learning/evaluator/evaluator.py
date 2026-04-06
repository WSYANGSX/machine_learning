from typing import TypeVar, Generic, Mapping, Any

from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms.base import AlgorithmBase

AlgoType = TypeVar("AlgoType", bound=AlgorithmBase)


class Evaluator(Generic[AlgoType]):
    def __init__(
        self,
        algo: AlgoType,
        ckpt: str,
        dataset: str | Mapping[str, Any],
        load_dataset: bool = True,
    ) -> None:
        self.ckpt = ckpt
        self._algorithm: AlgoType = algo
        self.load_dataset = load_dataset

        LOGGER.info("Algorithm initializing by evaluator...")
        self.algorithm._init_on_evaluator(self.ckpt, dataset, self.load_dataset)

    @property
    def algorithm(self) -> AlgoType:
        return self._algorithm

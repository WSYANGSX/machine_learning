from typing import TypeVar, Generic
from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms.base import AlgorithmBase


AlgoType = TypeVar("AlgoType", bound=AlgorithmBase)


class Predictor(Generic[AlgoType]):
    def __init__(self, algo: AlgoType, ckpt: str):
        self.ckpt = ckpt
        self.algorithm: AlgoType = algo

        LOGGER.info("Algorithm initializing by predictor...")
        self.algorithm._init_on_predictor(self.ckpt)

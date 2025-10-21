from typing import Any, Mapping
from machine_learning.utils import print_cfg
from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms import AlgorithmBase


class Evaluator:
    def __init__(
        self,
        ckpt: str,
        dataset: str | Mapping[str, Any],
        algo: AlgorithmBase,
    ) -> None:
        """
        The trainer of all machine learning algorithm

        Args:
            dataset (str, Mapping[str, Any]): The dataset cfg.
            algo (AlgorithmBase): The algorithm to be trained.
        """
        self.ckpt = ckpt
        self._algorithm = algo

        # ------------------- add cfg to algo -------------------------
        LOGGER.info("Algorithm initializing by evaluator...")
        self.algorithm._init_on_evaluator(self.ckpt, dataset)
        print_cfg("Total configuration", self.algorithm.cfg)

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def eval(self, *args, **kwargs) -> None:
        self.algorithm.eval(*args, **kwargs)

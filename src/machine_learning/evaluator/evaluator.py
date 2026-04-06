from typing import Any, Mapping
from machine_learning.utils import print_cfg
from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms import AlgorithmBase


class Evaluator:
    def __init__(
        self,
        algo: AlgorithmBase,
        ckpt: str,
        dataset: str | Mapping[str, Any],
        load_dataset: bool = True,
    ) -> None:
        """
        The evaluator of all machine learning algorithm.

        Args:
            algo (AlgorithmBase): The algorithm to be evaluated.
            ckpt (str): The checkpoint of the algorithm to be evaluat.
            dataset (str, Mapping[str, Any]): The dataset cfg.
            load_dataset (bool): Whether to load the test dataset provided by the evaluator. Defaults to True.
        """
        self.ckpt = ckpt
        self._algorithm = algo
        self.load_dataset = load_dataset

        # ------------------- add cfg to algo -------------------------
        LOGGER.info("Algorithm initializing by evaluator...")
        self.algorithm._init_on_evaluator(self.ckpt, dataset, self.load_dataset)
        print_cfg("Total configuration", self.algorithm.cfg)

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def eval(self, *args, **kwargs) -> None:
        self.algorithm.eval(*args, **kwargs)

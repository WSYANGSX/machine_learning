from typing import Mapping, Any, TypeVar

import json

from rich import box
from rich.json import JSON
from rich.table import Table
from rich.console import Console

from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms.base import AlgorithmBase

AlgoType = TypeVar("AlgoType", bound=AlgorithmBase)


class Evaluator:
    def __init__(
        self,
        algo: AlgoType,
        ckpt: str,
        dataset: str | Mapping[str, Any],
        load_dataset: bool = True,
    ) -> None:
        """Evaluator of all the algorithms."""
        self.ckpt = ckpt
        self._algorithm: AlgoType = algo
        self.load_dataset = load_dataset

        LOGGER.info("Algorithm initializing by evaluator...")
        self.algorithm._init_on_evaluator(self.ckpt, dataset, self.load_dataset)

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def eval(self) -> None:
        """
        Evaluate the preformece of the model on test dataset.

        Note:
            Verify the objective metrics (mAP, mIoU, Loss) of the model on the test set.
        """
        metrics, info = self.algorithm.eval()
        self.echo_info(metrics, info)

    def echo_info(self, metrics: dict[str, Any] | None = None, info: dict[Any] | None = None) -> None:
        """Echo the information of the evaluation."""
        console = Console()

        table = Table(
            title="Evaluation info Summary",
            header_style="bold magenta",
            show_header=True,
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold blue not italic",
        )

        table.add_column("Category", style="cyan", justify="center", vertical="middle")
        table.add_column("Item", style="dim", justify="center", vertical="middle")
        table.add_column("Value", justify="center", vertical="middle")

        def _format_value(val):
            if isinstance(val, (float, int)):
                # if the float is very small, use scientific notation
                if isinstance(val, float) and (val < 1e-4 and val > 0):
                    return f"{val:.4e}"
                if isinstance(val, float):
                    return f"{val:.8f}"
                return str(val)
            elif isinstance(val, (dict, list)):
                return JSON(json.dumps(val))
            else:
                return str(val)

        if metrics:
            for k, v in metrics.items():
                style = "bold green" if k == "save_best" else None
                val_str = _format_value(v)
                if style:
                    pass
                table.add_row("Eval Metric", k, val_str)

        if info:
            for k, v in info.items():
                table.add_row("Eval Info", k, _format_value(v))

        print("\n")
        console.print(table)

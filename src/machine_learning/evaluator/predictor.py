from typing import TypeVar, Generic, Mapping, Any, Literal

import torch
from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms import AlgorithmBase, global_factory
from machine_learning.utils import print_cfg

AlgoType = TypeVar("AlgoType", bound=AlgorithmBase)


class Predictor(Generic[AlgoType]):
    def __init__(
        self,
        ckpt: str,
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        algo_cfg = state["cfg"]
        name = algo_cfg["algorithm"]["name"]

        self.record_dir = algo_cfg["trainer"]["record_dir"]  # for save
        device = self._configure_device(device)

        # --------------------- build algorithm ---------------------
        self._build_algorithm(name, device, algo_cfg, False, False)
        self.algorithm._init_on_predictor(ckpt)
        print_cfg("Total configuration", algo_cfg)

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def _configure_device(self, device: str) -> torch.device:
        """Configure the trianing device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_algorithm(
        self,
        name: str,
        device: torch.device,
        cfg: Mapping[str, Any],
        amp: bool | None = None,
        ema: bool | None = False,
    ) -> None:
        LOGGER.info(f"Building algorithm {name}...")
        self._algorithm: AlgoType = global_factory.create_algorithm(
            algo=name, cfg=cfg, name=name, device=device, amp=amp, ema=ema
        )

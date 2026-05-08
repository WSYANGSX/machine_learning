from typing import Any, Optional
from copy import deepcopy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from machine_learning.utils.logger import LOGGER


class BaseNet(nn.Module, ABC):
    """
    Base class for all neural network models in this project.
    Provides common utilities, including weight initialization and model structure visualization.

    Notes:
        1. Subclasses must implement the 'dummy_input' property to enable structure visualization and FLOPs estimation.
        2. The 'view_structure' method relies on forward hooks to capture layer-wise output shapes and parameter counts
        during a dummy forward pass. Mixing 'nn.Module' components with functional APIs (e.g., 'torch.flatten', 'F.relu'
        ) will break the computation graph tracking. To ensure accurate structural tracing, strictly use 'nn.Module'
        layers registered in '__init__()' during the 'forward' pass.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        self.pretrained_path = pretrained_path

    @property
    def device(self) -> torch.device:
        """Returns the device on which the model's parameters are located."""
        return next(self.parameters()).device

    @property
    @abstractmethod
    def dummy_input(self) -> Any:
        """
        Returns a dummy input tensor (or tuple/dict of tensors) that can be used for model structure visualization and
        FLOPs estimation.

        Note:
            Subclasses must implement this property to enable these features.
        """
        pass

    def _init_weights(self):
        """Initialize weights of the model from pretrained ckpt or from scratch."""
        if self.pretrained_path:
            LOGGER.info(f"Loading pretrained weights for {self.__class__.__name__} from {self.pretrained_path}...")
            self._init_pretrained_weights(self.pretrained_path)

        else:
            LOGGER.info(f"Initializing weights of {self.__class__.__name__} for scratch...")
            self._init_scratch_weights()

    def _init_scratch_weights(self):
        """Initialize weights of the model from scratch."""
        for module in self.modules():
            if isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                    nn.Linear,
                ),
            ):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _init_pretrained_weights(
        self, ckpt: str, load_ema: bool = True, key: str | None = None, strict: bool = True
    ) -> None:
        """
        Initialize current single network from a pretrained checkpoint.

        Notes:
            1. This method belongs to BaseNet, so it only loads weights into self.
            2. If the checkpoint contains multiple networks, pass key to select one.
        """
        if not ckpt:
            return

        state = torch.load(ckpt, map_location=self.device, weights_only=False)

        def _select_from_mapping(mapping: dict, mapping_name: str):
            if key is not None:
                if key not in mapping:
                    raise KeyError(
                        f"Network key '{key}' not found in checkpoint['{mapping_name}']. "
                        f"Available keys: {list(mapping.keys())}"
                    )
                return key, mapping[key]

            if len(mapping) == 1:
                selected_key = next(iter(mapping.keys()))
                return selected_key, mapping[selected_key]

            raise ValueError(
                f"Checkpoint['{mapping_name}'] contains multiple networks: {list(mapping.keys())}. "
                f"Please pass key explicitly when loading pretrained weights."
            )

        # Case 1: checkpoint saved by AlgorithmBase with EMA
        if load_ema and isinstance(state, dict) and state.get("emas") is not None:
            selected_key, ema_state = _select_from_mapping(state["emas"], "emas")

            if isinstance(ema_state, dict) and "model_state" in ema_state:
                state_dict = ema_state["model_state"]
                LOGGER.info(f"Using EMA weights from checkpoint['emas']['{selected_key}'].")
            else:
                LOGGER.warning(f"Invalid EMA state for key '{selected_key}', falling back to normal network weights.")
                if "nets" not in state:
                    raise KeyError("Checkpoint has invalid EMA state and no normal 'nets' weights.")
                selected_key, state_dict = _select_from_mapping(state["nets"], "nets")

        # Case 2: checkpoint saved by AlgorithmBase without EMA
        elif isinstance(state, dict) and "nets" in state:
            selected_key, state_dict = _select_from_mapping(state["nets"], "nets")
            LOGGER.info(f"Using normal weights from checkpoint['nets']['{selected_key}'].")

        # Case 3: common checkpoint format: {'state_dict': ...}
        elif isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
            LOGGER.info("Using weights from checkpoint['state_dict'].")

        # Case 4: common checkpoint format: {'model_state': ...}
        elif isinstance(state, dict) and "model_state" in state:
            state_dict = state["model_state"]
            LOGGER.info("Using weights from checkpoint['model_state'].")

        # Case 5: raw state_dict
        else:
            state_dict = state
            LOGGER.info("Using checkpoint as a raw state_dict.")

        self.load_state_dict(state_dict, strict=strict)

        LOGGER.info(f"Successfully loaded pretrained weights into {self.__class__.__name__}.")

    def get_flops(self):
        """Return this model's FLOPs."""
        # Check if thop is available
        try:
            from thop import profile, clever_format

        except ImportError:
            raise ImportError("Please install 'thop' for FLOPs calculation: pip install thop.")

        if not hasattr(self, "dummy_input"):
            raise AttributeError(
                f"Model {self.__class__.__name__} has no 'dummy_input' attribute. "
                "Please define dummy_input property for FLOPs calculation."
            )

        dummy_input = self._prepare_dummy_input()
        try:
            # Note: The inputs parameter of thop needs to be a tuple.
            if isinstance(dummy_input, tuple):
                inputs_for_thop = dummy_input
            elif isinstance(dummy_input, dict):
                inputs_for_thop = tuple(dummy_input.values())
            else:
                inputs_for_thop = (dummy_input,)
        except Exception as e:
            raise ValueError(f"Failed to process dummy_input: {e}. dummy_input should be torch.Tensor, tuple, or dict.")

        # Calculate FLOPs
        try:
            macs, params = profile(
                deepcopy(self),
                inputs=inputs_for_thop,
                verbose=False,
                custom_ops={
                    nn.LayerNorm: None,
                    nn.GroupNorm: None,
                    nn.InstanceNorm1d: None,
                    nn.InstanceNorm2d: None,
                    nn.InstanceNorm3d: None,
                },
            )
            macs_formatted, _ = clever_format([macs, params], "%.3f")
            # Estimate FLOPs (MACs × 2)
            flops = macs * 2  # use real input size for FLOPs estimation
            flops_formatted, _ = clever_format([flops, params], "%.3f")

            return macs_formatted, flops_formatted

        except Exception as e:
            raise RuntimeError(
                f"FLOPs calculation failed: {e}. This could be due to unsupported operations or input format."
            )

    @staticmethod
    def _flatten_tensors(obj: Any) -> list[torch.Tensor]:
        """Recursively collect all tensors from a nested output object."""
        tensors: list[torch.Tensor] = []

        if isinstance(obj, torch.Tensor):
            tensors.append(obj)
        elif isinstance(obj, torch.nn.utils.rnn.PackedSequence):
            tensors.extend(BaseNet._flatten_tensors(obj.data))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                tensors.extend(BaseNet._flatten_tensors(item))
        elif isinstance(obj, dict):
            for item in obj.values():
                tensors.extend(BaseNet._flatten_tensors(item))

        return tensors

    @staticmethod
    def _shape_repr(obj: Any) -> Any:
        """Return a readable shape representation for tensors or nested tensor containers."""
        if isinstance(obj, torch.Tensor):
            return list(obj.shape)
        if isinstance(obj, torch.nn.utils.rnn.PackedSequence):
            return {"data": list(obj.data.shape)}
        if isinstance(obj, tuple):
            return tuple(BaseNet._shape_repr(item) for item in obj)
        if isinstance(obj, list):
            return [BaseNet._shape_repr(item) for item in obj]
        if isinstance(obj, dict):
            return {key: BaseNet._shape_repr(value) for key, value in obj.items()}
        return type(obj).__name__

    @staticmethod
    def _apply_to_tensors(obj: Any, fn) -> Any:
        """Apply fn to every tensor in a nested object while preserving the original container type."""
        if isinstance(obj, torch.Tensor):
            return fn(obj)
        if isinstance(obj, torch.nn.utils.rnn.PackedSequence):
            return obj._replace(data=fn(obj.data))
        if isinstance(obj, tuple):
            return tuple(BaseNet._apply_to_tensors(item, fn) for item in obj)
        if isinstance(obj, list):
            return [BaseNet._apply_to_tensors(item, fn) for item in obj]
        if isinstance(obj, dict):
            return {key: BaseNet._apply_to_tensors(value, fn) for key, value in obj.items()}
        return obj

    def _prepare_dummy_input(self) -> Any:
        """Get dummy_input and move all contained tensors to the model device."""
        try:
            dummy_input = self.dummy_input
        except Exception as e:
            raise AttributeError(
                f"Failed to get dummy_input for {self.__class__.__name__}: {e}. "
                "Please define a valid dummy_input property."
            ) from e

        try:
            device = self.device
        except StopIteration:
            return dummy_input

        return self._apply_to_tensors(dummy_input, lambda tensor: tensor.to(device))

    def _run_forward_with_dummy_input(self, dummy_input: Any) -> Any:
        """Run forward according to the type of dummy_input."""
        if isinstance(dummy_input, tuple):
            return self(*dummy_input)
        if isinstance(dummy_input, dict):
            return self(**dummy_input)
        return self(dummy_input)

    def _collect_structure_records(self) -> list[tuple[int, str, str, Any, int, list[int]]]:
        """Collect layer-wise structure records using forward hooks."""
        records: list[tuple[int, str, str, Any, int, list[int]]] = []
        tensor_to_idx: dict[int, int] = {}
        hooks: list[Any] = []
        container_types = (nn.ModuleDict, nn.ModuleList, nn.Sequential)

        def register_input_sources(obj: Any) -> None:
            input_tensors = self._flatten_tensors(obj)
            if len(input_tensors) == 1:
                tensor_to_idx[id(input_tensors[0])] = -1
            else:
                for i, tensor in enumerate(input_tensors):
                    tensor_to_idx[id(tensor)] = -1 - (i + 1)

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                idx = len(records)

                input_tensors = self._flatten_tensors(inputs)
                from_idxs = sorted(
                    {tensor_to_idx[id(tensor)] for tensor in input_tensors if id(tensor) in tensor_to_idx}
                )

                output_tensors = self._flatten_tensors(output)
                out_shape = self._shape_repr(output)
                n_params = sum(p.numel() for p in module.parameters())

                records.append((idx, name, module.__class__.__name__, out_shape, n_params, from_idxs))

                for tensor in output_tensors:
                    tensor_to_idx[id(tensor)] = idx

            return hook

        def register_structure_hooks(root: nn.Module, prefix: str = "") -> None:
            for child_name, child in root.named_children():
                full_name = child_name if not prefix else f"{prefix}.{child_name}"

                if isinstance(child, container_types) and root is self:
                    for sub_name, sub_module in child.named_children():
                        hooks.append(sub_module.register_forward_hook(make_hook(f"{full_name}.{sub_name}")))
                else:
                    hooks.append(child.register_forward_hook(make_hook(full_name)))

        training_states = {module: module.training for module in self.modules()}
        dummy_input = self._prepare_dummy_input()
        register_input_sources(dummy_input)
        register_structure_hooks(self)

        try:
            self.eval()
            with torch.no_grad():
                self._run_forward_with_dummy_input(dummy_input)
        finally:
            for hook in hooks:
                hook.remove()
            for module, was_training in training_states.items():
                module.train(was_training)

        return records

    @staticmethod
    def _format_from(from_idxs: list[int]) -> str:
        if not from_idxs:
            return "-1"
        if len(from_idxs) == 1:
            return str(from_idxs[0])
        return str(from_idxs)

    def _format_structure_records(self, records: list[tuple[int, str, str, Any, int, list[int]]]) -> str:
        """Format collected records as a readable table."""
        lines = [f"Model summary for {self.__class__.__name__}:"]

        if not records:
            lines.append("No records to display.")
            return "\n".join(lines)

        idxs, names, mtypes, shapes, n_params, froms = zip(*records)
        from_strs = [self._format_from(from_idx) for from_idx in froms]

        idx_len = max(max(len(str(idx)) for idx in idxs), 3)
        from_len = max(max(len(from_str) for from_str in from_strs), 4)
        name_len = max(max(len(str(name)) for name in names), 4) + 3
        mtype_len = max(max(len(str(mtype)) for mtype in mtypes), 4) + 3
        shape_len = max(max(len(str(shape)) for shape in shapes), 6) + 3
        param_len = max(max(len(str(param)) for param in n_params), 6) + 3

        lines.append(
            f"{'idx':>{idx_len}} {'from':>{from_len}} {'name':>{name_len}} {'type':>{mtype_len}}"
            f" {'output':>{shape_len}} {'params':>{param_len}}"
        )

        for (idx, name, mtype, shape, n_param, _), from_str in zip(records, from_strs):
            lines.append(
                f"{idx:>{idx_len}} {from_str:>{from_len}} {name:>{name_len}} {mtype:>{mtype_len}}"
                f" {str(shape):>{shape_len}} {n_param:>{param_len}d}"
            )

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(
            f"Total params: {total_params:,} | Trainable params: {trainable:,}"
            f" | Non-trainable params: {total_params - trainable:,}"
        )

        try:
            macs, flops = self.get_flops()
            lines.append(f"MACs (Multiply-Accumulates): {macs} | FLOPs (estimated): {flops}")
        except ImportError as e:
            lines.append(f"{e} Skipping FLOPs calculation.")
        except (AttributeError, ValueError, RuntimeError) as e:
            lines.append(f"FLOPs calculation skipped: {e}")
        except Exception as e:
            lines.append(f"Unexpected error during FLOPs calculation: {e}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a formatted model structure summary."""
        records = self._collect_structure_records()
        return self._format_structure_records(records)

    def view_structure(self) -> None:
        """Print the model structure summary."""
        LOGGER.info(self.summary())

    def __str__(self) -> str:
        """String representation of the model structure summary."""
        return self.summary()

    def __repr__(self):
        return self.__str__()

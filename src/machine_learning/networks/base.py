from typing import Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from machine_learning.utils.logger import LOGGER


class BaseNet(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    @abstractmethod
    def dummy_input(self) -> Any:
        pass

    def _initialize_weights(self):
        LOGGER.info(f"Initializing weights of {self.__class__.__name__} with Kaiming normal...")

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
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def view_structure(self):
        LOGGER.info(f"Model summary for {self.__class__.__name__}:")

        # 检查thop是否可用
        try:
            from thop import profile, clever_format
            from copy import deepcopy

            thop_available = True
        except ImportError:
            thop_available = False
            LOGGER.warning("Please install 'thop' for FLOPs calculation: pip install thop")

        records = []
        tensor_to_idx: dict[int, int] = {}  # id(tensor) -> layer_idx

        def collect_tensor_ids(obj):
            """Recursively collect the ids of all tensors in obj"""
            ids = []
            if isinstance(obj, torch.Tensor):
                ids.append(id(obj))
            elif isinstance(obj, (list, tuple)):
                for o in obj:
                    ids.extend(collect_tensor_ids(o))
            elif isinstance(obj, dict):
                for o in obj.values():
                    ids.extend(collect_tensor_ids(o))
            elif isinstance(obj, torch.nn.utils.rnn.PackedSequence):
                # Handling special cases of RNN
                ids.extend(collect_tensor_ids(obj.data))
            return ids

        def register_output_tensors(output):
            """Register all the output tensors to tensor to idx"""
            if isinstance(output, torch.Tensor):
                return [id(output)]
            elif isinstance(output, (list, tuple)):
                ids = []
                for o in output:
                    ids.extend(register_output_tensors(o))
                return ids
            elif isinstance(output, dict):
                ids = []
                for o in output.values():
                    ids.extend(register_output_tensors(o))
                return ids
            return []

        def make_hook(name):
            def hook(module, inputs, output):
                idx = len(records)

                # Processes the input source
                in_ids = []
                for inp in inputs:
                    in_ids.extend(collect_tensor_ids(inp))

                from_idxs = []
                for tid in in_ids:
                    if tid in tensor_to_idx:
                        from_idxs.append(tensor_to_idx[tid])
                from_idxs = sorted(set(from_idxs))

                # Process the output
                output_tensors = []

                if isinstance(output, torch.Tensor):
                    out_shape = list(output.shape)
                    output_tensors = [output]
                elif isinstance(output, (list, tuple)):
                    # Flattening collects all tensors
                    def flatten_outputs(obj):
                        tensors = []
                        if isinstance(obj, torch.Tensor):
                            tensors.append(obj)
                        elif isinstance(obj, (list, tuple)):
                            for item in obj:
                                tensors.extend(flatten_outputs(item))
                        elif isinstance(obj, dict):
                            for item in obj.values():
                                tensors.extend(flatten_outputs(item))
                        return tensors

                    output_tensors = flatten_outputs(output)

                    # Generate shape descriptions
                    if all(isinstance(t, torch.Tensor) for t in output_tensors):
                        out_shape = [list(t.shape) for t in output_tensors]
                    else:
                        out_shape = str(type(output))
                else:
                    out_shape = str(type(output))

                # Number of Parameters
                n_params = sum(p.numel() for p in module.parameters())

                # Record the current layer information
                records.append((idx, name, module.__class__.__name__, out_shape, n_params, from_idxs))

                # Register all output tensors to the dictionary
                for tensor in output_tensors:
                    if isinstance(tensor, torch.Tensor):
                        tensor_to_idx[id(tensor)] = idx

            return hook

        hooks = []
        container_types = (nn.ModuleDict, nn.ModuleList, nn.Sequential)

        def register_big_blocks(root: nn.Module, prefix: str = ""):
            for child_name, child in root.named_children():
                full_name = f"{prefix}{child_name}" if prefix == "" else f"{prefix}.{child_name}"

                if isinstance(child, container_types):
                    # Container type delves into it only when it is a direct submodule of self
                    if root is self:
                        for k, m in child.named_children():
                            block_name = f"{full_name}.{k}"
                            hooks.append(m.register_forward_hook(make_hook(block_name)))
                    else:
                        # For non-top-level containers, directly register the container itself
                        hooks.append(child.register_forward_hook(make_hook(full_name)))
                else:
                    # Non-container modules, register directly
                    hooks.append(child.register_forward_hook(make_hook(full_name)))

        register_big_blocks(self)

        # Initialize the source of the input tensor
        if hasattr(self, "dummy_input"):
            dummy_input = self.dummy_input
            if isinstance(dummy_input, torch.Tensor):
                tensor_to_idx[id(dummy_input)] = -1  # Input marked as -1

            elif isinstance(dummy_input, (list, tuple)):
                for i, inp in enumerate(dummy_input):
                    if isinstance(inp, torch.Tensor):
                        tensor_to_idx[id(inp)] = -1 - (i + 1)  # Mark multiple inputs separately

        # Run forward propagation
        self.eval()
        with torch.no_grad():
            if hasattr(self, "dummy_input"):
                dummy_input = self.dummy_input
                if isinstance(dummy_input, tuple):
                    _ = self.forward(*dummy_input)
                elif isinstance(dummy_input, dict):
                    _ = self.forward(**dummy_input)
                else:
                    _ = self.forward(dummy_input)
            else:
                LOGGER.warning("Model has no dummy_input attribute, skipping forward pass")
                return records

        # Remove Hooks
        for h in hooks:
            h.remove()

        # Calculate alignment width
        if records:
            idxs, names, mtypes, shapes, n_params, froms = zip(*records)
            idx_len = max(max(len(str(id)) for id in idxs), 3)
            from_len = max(len(str(f)) for f in froms) if froms else 5
            name_len = max(len(str(name)) for name in names) + 3
            mtype_len = max(len(str(mtype)) for mtype in mtypes) + 3
            shape_len = max(len(str(shape)) for shape in shapes) + 3
            param_len = max(len(str(param)) for param in n_params) + 3

            # Print the table header
            print(
                f"{'idx':>{idx_len}} {'from':>{from_len}} {'name':>{name_len}} {'type':>{mtype_len}}"
                f" {'output':>{shape_len}} {'params':>{param_len}}"
            )

            # Print Record
            for idx, name, mtype, shape, n_param, f in records:
                # Format the from field
                from_str = "-1" if not f else str(f[0]) if len(f) == 1 else str(f)
                print(
                    f"{idx:>{idx_len}} {from_str:>{from_len}} {name:>{name_len}} {mtype:>{mtype_len}}"
                    f" {str(shape):>{shape_len}} {n_param:>{param_len}d}"
                )

        else:
            print("No records to display.")
            return records

        # Statistics of total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Safely calculate FLOPs
        if thop_available and hasattr(self, "dummy_input"):
            try:
                # Note: The inputs parameter of thop needs to be a tuple.
                dummy_input = self.dummy_input
                if isinstance(dummy_input, tuple):
                    inputs_for_thop = dummy_input
                elif isinstance(dummy_input, dict):
                    inputs_for_thop = tuple(dummy_input.values())
                else:
                    inputs_for_thop = (dummy_input,)

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
                macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
                # 估算FLOPs（MACs × 2）
                flops = macs * 2
                flops_formatted, _ = clever_format([flops, params], "%.3f")

                print(f"\nTotal params: {total_params:,}")
                print(f"Trainable params: {trainable:,}")
                print(f"Non-trainable params: {total_params - trainable:,}")
                print(f"MACs (Multiply-Accumulates): {macs_formatted}")
                print(f"FLOPs (estimated): {flops_formatted}.")

            except Exception as e:
                print(f"\nTotal params: {total_params:,}, Trainable params: {trainable:,}")
                print(f"MACs/FLOPs calculation failed: {e}.")

        else:
            print(f"\nTotal params: {total_params:,}, Trainable params: {trainable:,}")
            if not thop_available:
                print("Install 'thop' for MACs calculation: pip install thop.")

import torch
import torch.nn as nn


def attach_nan_forward_hooks(model: nn.Module, print_first_only: bool = True):
    """
    Register forward hooks for all sub-modules of the model. Once NaN/Inf appears in the output of a certain layer,
    print and throw an exception.
    """
    hooks = []
    triggered = {"flag": False}  # Wrap it with a dict for easy modification within a closure

    def _hook_fn(module, inputs, outputs):
        if triggered["flag"] and print_first_only:
            return

        def _check_tensor(t):
            if not torch.is_tensor(t):
                return False
            if t.dtype.is_floating_point:
                return not torch.isfinite(t).all()
            return False

        has_bad = False
        if isinstance(outputs, (tuple, list)):
            for o in outputs:
                if _check_tensor(o):
                    has_bad = True
                    break
        else:
            has_bad = _check_tensor(outputs)

        if has_bad:
            triggered["flag"] = True
            print(
                f"\n[NaN FORWARD] module={module.__class__.__name__} full_name={getattr(module, '_full_name', 'N/A')}"
            )
            # print out the output statistics
            if isinstance(outputs, torch.Tensor):
                print(
                    f"output stats: min={outputs.min().item():.4e}, "
                    f"max={outputs.max().item():.4e}, "
                    f"mean={outputs.mean().item():.4e}"
                )
            raise RuntimeError("NaN detected in forward pass")

    # Register hooks for all sub-modules and remember their "full names"
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:  # Only attach hooks to leaf modules to reduce redundancy
            m._full_name = name
            hooks.append(m.register_forward_hook(_hook_fn))

    return hooks


def attach_nan_backward_hooks(model: nn.Module, print_first_only: bool = True):
    hooks = []
    triggered = {"flag": False}

    def _bw_hook(module, grad_input, grad_output):
        if triggered["flag"] and print_first_only:
            return

        def _has_bad(t):
            return torch.is_tensor(t) and t.dtype.is_floating_point and not torch.isfinite(t).all()

        bad = False
        for g in grad_output:
            if _has_bad(g):
                bad = True
                break

        if bad:
            triggered["flag"] = True
            print(
                f"\n[NaN BACKWARD] module={module.__class__.__name__} full_name={getattr(module, '_full_name', 'N/A')}"
            )
            raise RuntimeError("NaN detected in backward pass")

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            m._full_name = name
            hooks.append(m.register_full_backward_hook(_bw_hook))

    return hooks

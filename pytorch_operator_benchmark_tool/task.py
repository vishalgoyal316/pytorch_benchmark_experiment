import torch
import torch.nn.functional as F


TASK_DIMENSIONS = {"matmul": 3, "softmax": 2, "layer_norm": 2, "linear_gelu": 3, "pointwise_fusion": 2}


def matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.matmul(left, right)


def softmax(input: torch.Tensor) -> torch.Tensor:
    return torch.softmax(input, dim=-1)


def layer_norm(input: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(input, input.shape[-1:])


def linear_gelu(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(F.linear(input, weight, bias))


def pointwise_fusion(input: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.silu((input + bias) * 0.5 + residual), -10, 10)


def parse_task_shape(operation: str, value: str) -> tuple[int, ...]:
    expected_dimensions = TASK_DIMENSIONS.get(operation)
    if expected_dimensions is None:
        raise ValueError(f"unsupported operation {operation!r}")
    try:
        dimensions = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError(f"invalid {operation} shape {value!r}") from error
    if len(dimensions) != expected_dimensions:
        raise ValueError(f"invalid {operation} shape {value!r}; expected {expected_dimensions} dimensions")
    if min(dimensions) <= 0:
        raise ValueError(f"invalid {operation} shape {value!r}; dimensions must be positive")
    return dimensions

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, type]
    required: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.required is None:
            return
        if len(set(self.required)) != len(self.required):
            raise ValueError("required parameter names must not be duplicated")
        unknown = set(self.required) - set(self.parameters)
        if unknown:
            raise ValueError(f"required parameter names are unknown: {', '.join(sorted(unknown))}")


def validate_tool_arguments(tool: Tool, raw_args: dict) -> dict:
    if not isinstance(raw_args, dict):
        raise ValueError("tool arguments must be an object")
    validated: dict = {}
    required = tool.required if tool.required is not None else tuple(tool.parameters)
    for name in required:
        if name not in raw_args:
            raise ValueError(f"missing required argument: {name}")
    for name, expected_type in tool.parameters.items():
        if name not in raw_args:
            continue
        value = raw_args[name]
        if not isinstance(value, expected_type):
            raise ValueError(f"{name} must be {expected_type.__name__}")
        validated[name] = value
    unexpected = set(raw_args) - set(tool.parameters)
    if unexpected:
        raise ValueError(f"unexpected arguments: {', '.join(sorted(unexpected))}")
    return validated

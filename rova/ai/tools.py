from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jsonschema


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, type] = field(default_factory=dict)
    required: tuple[str, ...] | None = None
    input_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.input_schema is not None:
            if self.parameters or self.required is not None:
                raise ValueError("input_schema cannot be combined with legacy parameters")
            if not isinstance(self.input_schema, dict) or self.input_schema.get("type") != "object":
                raise ValueError("input_schema must be an object JSON Schema")
            return
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
    if tool.input_schema is not None:
        try:
            jsonschema.validate(raw_args, tool.input_schema)
        except jsonschema.ValidationError as error:
            if error.validator == "enum" and error.path:
                field = str(error.path[-1])
                allowed = error.validator_value
                if isinstance(allowed, list) and all(isinstance(item, str) for item in allowed):
                    raise ValueError(
                        f"Invalid {field}: {error.instance!r}. Expected one of: {', '.join(allowed)}."
                    ) from error
            raise ValueError(f"tool argument validation failed: {error.message}") from error
        return dict(raw_args)
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

import argparse
from collections.abc import Sequence


def convert_value(value: str) -> int | float | bool | str:
    """Convert a string value to int, float, or bool when appropriate."""
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    lower_value = value.lower()
    if lower_value in ("true", "false"):
        return lower_value == "true"

    return value


class KeyValueParserAction(argparse.Action):
    """Custom argparse action to parse key=value pairs and convert values."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str],
        option_string: str | None = None,
    ) -> None:
        """Parse key=value pairs and convert values to int, float, or bool."""
        if values is None:
            return
        if isinstance(values, str):
            values = [values]

        result = {}
        for pair in values:
            try:
                key, value = pair.split("=")
                result[key] = convert_value(value)
            except ValueError as e:
                message = f"Error on '{pair}' - it should be in key=value format.\nTraceback: {e}"
                raise argparse.ArgumentError(self, message)
        setattr(namespace, self.dest, result)

from typing import Any


def class_name(obj: Any) -> str:
    if isinstance(obj, type):
        # obj is a class
        return obj.__name__
    else:
        # obj is an instance
        return type(obj).__name__


def full_class_name(obj: Any) -> str:
    """Get fully qualified class name including module path."""
    if isinstance(obj, type):
        # obj is a class
        cls = obj
    else:
        # obj is an instance
        cls = type(obj)

    module = cls.__module__
    qualname = cls.__qualname__

    # Handle built-ins (module is 'builtins')
    if module == "builtins":
        return qualname

    return f"{module}.{qualname}"


def cli_sanitise(value: str) -> str:
    return value.replace("_", "-")

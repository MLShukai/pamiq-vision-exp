type size_2d = int | tuple[int, int]


def size_2d_to_int_tuple(size: size_2d) -> tuple[int, int]:
    """Convert `size_2d` type to int tuple."""
    return (size, size) if isinstance(size, int) else (size[0], size[1])


def get_class_module_path(cls: type) -> str:
    """Get the module path of a class.

    Args:
        cls: The class to get the module path.

    Returns:
        str: The module path of the class.
    """
    return f"{cls.__module__}.{cls.__name__}"

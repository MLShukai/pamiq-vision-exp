type size_3d = int | tuple[int, int, int]


def size_3d_to_tuple(size: size_3d) -> tuple[int, int, int]:
    if isinstance(size, int):
        return size, size, size
    return size

import pytest

from exp.utils import Size2D, size_2d_to_int_tuple


@pytest.mark.parametrize("input,expected", [(10, (10, 10)), ((2, 3), (2, 3))])
def test_size_2d_to_int_tuple(input: Size2D, expected: tuple[int, int]):
    assert size_2d_to_int_tuple(input) == expected

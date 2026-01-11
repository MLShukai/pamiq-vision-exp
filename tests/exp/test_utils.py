import pytest

from exp.utils import size_3d_to_tuple


class TestSize3dToTuple:
    @pytest.mark.parametrize(
        "size, expected",
        [
            (1, (1, 1, 1)),
            (5, (5, 5, 5)),
            ((1, 2, 3), (1, 2, 3)),
            ((10, 20, 30), (10, 20, 30)),
        ],
    )
    def test_size_3d_to_tuple(self, size, expected):
        assert size_3d_to_tuple(size) == expected

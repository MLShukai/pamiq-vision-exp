from exp.oc_resolvers import compute_feature_size


class TestComputeFeatureSize:
    def test_basic_computation(self):
        # n_tubelets = (8, 14, 14), feature_size = 8*14*14*128 = 200704
        result = compute_feature_size([16, 224, 224], [2, 16, 16], 128)
        assert result == 8 * 14 * 14 * 128

    def test_different_dimensions(self):
        result = compute_feature_size([8, 112, 112], [2, 16, 16], 64)
        assert result == 4 * 7 * 7 * 64

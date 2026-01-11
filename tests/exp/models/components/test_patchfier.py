import pytest
import torch

from exp.models.components.patchfier import VideoPatchDecoder, VideoPatchifier


class TestVideoPatchifier:
    @pytest.mark.parametrize(
        "batch_size, channels, time, height, width, tubelet_size, expected_n_tubelets",
        [
            (2, 3, 16, 224, 224, (2, 16, 16), (8, 14, 14)),
            (1, 3, 8, 112, 112, (2, 16, 16), (4, 7, 7)),
            (4, 3, 32, 160, 160, (2, 16, 16), (16, 10, 10)),
        ],
    )
    def test_forward(
        self,
        batch_size,
        channels,
        time,
        height,
        width,
        tubelet_size,
        expected_n_tubelets,
    ):
        patchifier = VideoPatchifier(
            tubelet_size=tubelet_size, in_channels=channels, embed_dim=768
        )

        video = torch.randn(batch_size, channels, time, height, width)
        output = patchifier(video)

        n_t, n_h, n_w = expected_n_tubelets
        expected_n = n_t * n_h * n_w

        assert output.shape == (batch_size, expected_n, 768)

    @pytest.mark.parametrize(
        "video_shape, tubelet_size, expected",
        [
            ((16, 224, 224), (2, 16, 16), (8, 14, 14)),
            ((8, 112, 112), (2, 16, 16), (4, 7, 7)),
            ((32, 160, 160), (2, 16, 16), (16, 10, 10)),
            ((16, 128, 128), 16, (1, 8, 8)),
        ],
    )
    def test_compute_num_tubelets(self, video_shape, tubelet_size, expected):
        result = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)
        assert result == expected

    @pytest.mark.parametrize(
        "video_shape, tubelet_size",
        [
            ((1, 224, 224), (2, 16, 16)),
            ((16, 15, 224), (2, 16, 16)),
            ((16, 224, 15), (2, 16, 16)),
        ],
    )
    def test_compute_num_tubelets_raises_on_small_video(
        self, video_shape, tubelet_size
    ):
        with pytest.raises(ValueError, match="too small"):
            VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)


class TestVideoPatchDecoder:
    @pytest.mark.parametrize(
        "batch_size, n_tubelets, tubelet_size, embed_dim, out_channels",
        [
            (2, (8, 14, 14), (2, 16, 16), 768, 3),
            (1, (4, 7, 7), (2, 16, 16), 384, 3),
            (4, (16, 10, 10), (2, 16, 16), 512, 1),
        ],
    )
    def test_forward(
        self, batch_size, n_tubelets, tubelet_size, embed_dim, out_channels
    ):
        decoder = VideoPatchDecoder(
            n_tubelets=n_tubelets,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            out_channels=out_channels,
        )

        n_t, n_h, n_w = n_tubelets
        total_tubelets = n_t * n_h * n_w
        latents = torch.randn(batch_size, total_tubelets, embed_dim)

        output = decoder(latents)

        expected_time = n_t * tubelet_size[0]
        expected_height = n_h * tubelet_size[1]
        expected_width = n_w * tubelet_size[2]

        assert output.shape == (
            batch_size,
            out_channels,
            expected_time,
            expected_height,
            expected_width,
        )

    def test_forward_raises_on_wrong_num_tubelets(self):
        decoder = VideoPatchDecoder(
            n_tubelets=(8, 14, 14), tubelet_size=(2, 16, 16), embed_dim=768
        )

        wrong_latents = torch.randn(2, 100, 768)

        with pytest.raises(ValueError, match="expected"):
            decoder(wrong_latents)

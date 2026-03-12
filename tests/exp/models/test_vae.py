import torch

from exp.models.vae import SimpleVAE


class TestSimpleVAE:
    def test_forward_shape(self):
        vae = SimpleVAE(in_channels=3, latent_dim=64)
        x = torch.randn(2, 3, 32, 32)
        recon, mu, log_var = vae(x)
        assert recon.shape == (2, 3, 32, 32)
        assert mu.shape == (2, 64)
        assert log_var.shape == (2, 64)

    def test_encode_shape(self):
        vae = SimpleVAE(in_channels=3, latent_dim=128)
        x = torch.randn(4, 3, 32, 32)
        mu, log_var = vae.encode(x)
        assert mu.shape == (4, 128)
        assert log_var.shape == (4, 128)

    def test_decode_shape(self):
        vae = SimpleVAE(in_channels=3, latent_dim=64)
        z = torch.randn(2, 64)
        out = vae.decode(z)
        assert out.shape == (2, 3, 32, 32)

    def test_gradient_flows(self):
        vae = SimpleVAE(in_channels=3, latent_dim=32)
        x = torch.randn(2, 3, 32, 32)
        recon, mu, log_var = vae(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        for p in vae.parameters():
            assert p.grad is not None

import torch

from exp.models.vae import VAEDecoder, VAEEncoder, create_vae


class TestVAEEncoder:
    def test_forward_shape(self):
        latent_dim = 512
        encoder = VAEEncoder(latent_dim=latent_dim, in_channels=3)
        video = torch.randn(2, 3, 4, 32, 32)
        out = encoder(video)
        assert out.shape == (2, latent_dim)

    def test_encode_with_logvar_shape(self):
        latent_dim = 512
        encoder = VAEEncoder(latent_dim=latent_dim)
        video = torch.randn(2, 3, 4, 32, 32)
        mu, log_var = encoder.encode_with_logvar(video)
        assert mu.shape == (2, latent_dim)
        assert log_var.shape == (2, latent_dim)

    def test_reparameterize_shape(self):
        latent_dim = 512
        encoder = VAEEncoder(latent_dim=latent_dim)
        mu = torch.randn(2, latent_dim)
        log_var = torch.randn(2, latent_dim)
        z = encoder.reparameterize(mu, log_var)
        assert z.shape == (2, latent_dim)


class TestVAEDecoder:
    def test_forward_shape_flat_input(self):
        latent_dim = 512
        decoder = VAEDecoder(latent_dim=latent_dim, in_channels=3)
        z = torch.randn(2, latent_dim)
        out = decoder(z)
        assert out.shape == (2, 3, 32, 32)


class TestCreateVAE:
    def test_returns_encoder_and_decoder(self):
        models = create_vae(
            video_shape=(4, 32, 32),
            tubelet_size=(2, 8, 8),
            embed_dim=16,
        )
        assert "encoder" in models
        assert "decoder" in models
        assert isinstance(models["encoder"], VAEEncoder)
        assert isinstance(models["decoder"], VAEDecoder)

    def test_latent_dim_from_config(self):
        models = create_vae(
            video_shape=(4, 32, 32),
            tubelet_size=(2, 8, 8),
            embed_dim=16,
        )
        encoder = models["encoder"]
        video = torch.randn(1, 3, 4, 32, 32)
        out = encoder(video)
        # n_tubelets = (4//2, 32//8, 32//8) = (2, 4, 4), latent_dim = 2*4*4*16 = 512
        assert out.shape == (1, 512)

    def test_gradient_flows(self):
        models = create_vae(
            video_shape=(4, 32, 32),
            tubelet_size=(2, 8, 8),
            embed_dim=16,
        )
        encoder = models["encoder"]
        decoder = models["decoder"]
        video = torch.randn(2, 3, 4, 32, 32)
        mu, log_var = encoder.encode_with_logvar(video)
        z = encoder.reparameterize(mu, log_var)
        recon = decoder(z)
        loss = recon.sum()
        loss.backward()
        for p in encoder.parameters():
            assert p.grad is not None
        for p in decoder.parameters():
            assert p.grad is not None

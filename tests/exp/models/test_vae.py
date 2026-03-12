import torch

from exp.models.vae import VAEDecoder, VAEEncoder, create_vae


class TestVAEEncoder:
    def test_forward_shape(self):
        n_tubelets = (2, 4, 4)
        embed_dim = 16
        encoder = VAEEncoder(n_tubelets=n_tubelets, embed_dim=embed_dim, in_channels=3)
        video = torch.randn(2, 3, 4, 32, 32)
        out = encoder(video)
        assert out.shape == (2, 2 * 4 * 4, embed_dim)

    def test_forward_ignores_masks(self):
        n_tubelets = (2, 4, 4)
        embed_dim = 16
        encoder = VAEEncoder(n_tubelets=n_tubelets, embed_dim=embed_dim)
        video = torch.randn(2, 3, 4, 32, 32)
        masks = torch.ones(2, 32, dtype=torch.bool)
        out_no_mask = encoder(video)
        out_with_mask = encoder(video, masks)
        assert torch.equal(out_no_mask, out_with_mask)

    def test_encode_with_logvar_shape(self):
        n_tubelets = (2, 4, 4)
        embed_dim = 16
        encoder = VAEEncoder(n_tubelets=n_tubelets, embed_dim=embed_dim)
        video = torch.randn(2, 3, 4, 32, 32)
        mu, log_var = encoder.encode_with_logvar(video)
        latent_dim = 2 * 4 * 4 * embed_dim
        assert mu.shape == (2, latent_dim)
        assert log_var.shape == (2, latent_dim)

    def test_reparameterize_shape(self):
        n_tubelets = (2, 4, 4)
        embed_dim = 16
        encoder = VAEEncoder(n_tubelets=n_tubelets, embed_dim=embed_dim)
        latent_dim = 2 * 4 * 4 * embed_dim
        mu = torch.randn(2, latent_dim)
        log_var = torch.randn(2, latent_dim)
        z = encoder.reparameterize(mu, log_var)
        assert z.shape == (2, latent_dim)


class TestVAEDecoder:
    def test_forward_shape_flat_input(self):
        n_tubelets = (2, 4, 4)
        embed_dim = 16
        decoder = VAEDecoder(n_tubelets=n_tubelets, embed_dim=embed_dim, in_channels=3)
        latent_dim = 2 * 4 * 4 * embed_dim
        z = torch.randn(2, latent_dim)
        out = decoder(z)
        assert out.shape == (2, 3, 32, 32)

    def test_forward_shape_3d_input(self):
        n_tubelets = (2, 4, 4)
        embed_dim = 16
        decoder = VAEDecoder(n_tubelets=n_tubelets, embed_dim=embed_dim, in_channels=3)
        z = torch.randn(2, 2 * 4 * 4, embed_dim)
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

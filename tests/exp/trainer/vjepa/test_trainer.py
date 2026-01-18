from functools import partial

import torch
from pamiq_core.data.impls import RandomReplacementBuffer
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel

from exp.models.vjepa import Encoder, Predictor
from exp.trainer.vjepa.collator import VideoMultiBlockMaskCollator
from exp.trainer.vjepa.trainer import VJEPATrainer
from tests.helpers import parametrize_device


class TestVJEPATrainer:
    @parametrize_device
    def test_run(self, device: torch.device):
        n_tubelets = (2, 4, 4)
        hidden_dim = 32
        embed_dim = 16
        batch_size = 2
        data_size = 4

        context_encoder = Encoder(
            patchifier=None,
            n_tubelets=n_tubelets,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            depth=1,
            num_heads=2,
        )
        target_encoder = context_encoder.clone()
        predictor = Predictor(
            n_tubelets=n_tubelets,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            depth=1,
            num_heads=2,
        )

        models = {
            "context_encoder": TorchTrainingModel(context_encoder, device=device),
            "target_encoder": TorchTrainingModel(
                target_encoder, has_inference_model=False, device=device
            ),
            "predictor": TorchTrainingModel(
                predictor, has_inference_model=False, device=device
            ),
        }

        buffer: RandomReplacementBuffer[torch.Tensor] = RandomReplacementBuffer(
            max_size=data_size
        )
        n_tubelets_total = n_tubelets[0] * n_tubelets[1] * n_tubelets[2]
        for _ in range(data_size):
            buffer.add(torch.randn(n_tubelets_total, hidden_dim))

        collator = VideoMultiBlockMaskCollator(num_tubelets=n_tubelets)

        trainer = VJEPATrainer(
            partial_optimizer=partial(torch.optim.Adam, lr=1e-3),
            collate_fn=collator,
            context_encoder_name="context_encoder",
            target_encoder_name="target_encoder",
            predictor_name="predictor",
            data_user_name="video",
            batch_size=batch_size,
            max_epochs=1,
        )

        connect_components(
            trainers=trainer,
            buffers={"video": buffer},
            models=models,
        )

        trainer.run()

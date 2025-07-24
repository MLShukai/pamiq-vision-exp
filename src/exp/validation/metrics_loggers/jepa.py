from typing import override

import hydra
import torch
from pamiq_core.torch import get_device
from torch.utils.data import DataLoader

from exp.aim_utils import get_global_run
from exp.models import ModelName
from exp.models.jepa import Encoder, Predictor
from exp.trainers.jepa import JEPATrainer

from .base import MetricsLogger


class JEPAMetrics(MetricsLogger):
    """Metrics logger for JEPA (Joint Embedding Predictive Architecture)
    models.

    This class computes and logs per-sample losses for JEPA models on a
    validation dataset. It evaluates how well the predictor can
    reconstruct target encoder representations from masked context
    encoder representations.
    """

    def __init__(self, batch_size: int, log_prefix: str = "jepa") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.log_prefix = log_prefix

    @override
    @torch.inference_mode()
    def run(self) -> None:
        # Validate Aim run is initialized
        if (aim_run := get_global_run()) is None:
            raise ValueError(
                "Aim run not initialized. Please set global aim run before calling JEPAMetrics."
            )

        # Get required models from registry
        context_encoder = self.models[ModelName.JEPA_CONTEXT_ENCODER]
        target_encoder = self.models[ModelName.JEPA_TARGET_ENCODER]
        predictor = self.models[ModelName.JEPA_PREDICTOR]
        # Validate model types
        if not (
            isinstance(context_encoder, Encoder)
            and isinstance(target_encoder, Encoder)
            and isinstance(predictor, Predictor)
        ):
            raise ValueError(
                f"Invalid model types. Expected (Encoder, Encoder, Predictor), "
                f"got ({type(context_encoder).__name__}, {type(target_encoder).__name__}, "
                f"{type(predictor).__name__})"
            )

        # Determine computation device from context encoder
        device = get_device(context_encoder)

        # Setup DataLoader with JEPA-specific collate function
        collate_fn = hydra.utils.instantiate(self.exp_cfg.trainer.jepa.collate_fn)
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,  # No shuffling for consistent metrics
            drop_last=False,  # Process all samples
        )

        # Process batches and compute per-sample losses
        index = 0
        for batch in dataloader:
            # Move batch tensors to computation device
            batch = tuple(map(lambda x: x.to(device), batch))

            # Compute JEPA loss for this batch
            loss_dict = JEPATrainer.compute_loss(
                target_encoder,
                context_encoder,
                predictor,
                batch,
            )

            # Log individual sample losses
            for loss in loss_dict["loss_per_data"]:
                aim_run.track(
                    loss.cpu().item(),
                    "loss",
                    step=index,
                    context={"namespace": "metrics", "metrics_type": self.log_prefix},
                )
                index += 1  # Increment index for each sample

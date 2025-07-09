from typing import override

from pamiq_core import Agent
from torch import Tensor

from exp.data import BufferNames, DataKeys


class ImageCollectingAgent(Agent[Tensor, None]):
    """Agent that collects image observations for vision experiments."""

    @override
    def on_data_collectors_attached(self) -> None:
        """Set up the image data collector when attached to the agent."""
        super().on_data_collectors_attached()
        self.collector = self.get_data_collector(BufferNames.IMAGE)

    @override
    def step(self, observation: Tensor) -> None:
        """Collect the observed image tensor.

        Args:
            observation: Image tensor to collect
        """
        self.collector.collect({DataKeys.IMAGE: observation})
        return

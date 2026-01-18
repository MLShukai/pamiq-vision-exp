import pytest
import torch
from pamiq_core.data.impls import RandomReplacementBuffer
from pamiq_core.testing import connect_components
from torch import Tensor

from exp.agents.frame_stacking import FrameStackingAgent


class TestFrameStackingAgent:
    """Test suite for FrameStackingAgent."""

    @pytest.mark.parametrize("num_frames", [0, -1])
    def test_init_validates_num_frames(self, num_frames: int):
        """Test that num_frames must be positive."""
        with pytest.raises(ValueError, match="num_frames must be positive"):
            FrameStackingAgent(num_frames=num_frames, frame_stride=1)

    @pytest.mark.parametrize("frame_stride", [0, -1])
    def test_init_validates_frame_stride(self, frame_stride: int):
        """Test that frame_stride must be positive."""
        with pytest.raises(ValueError, match="frame_stride must be positive"):
            FrameStackingAgent(num_frames=4, frame_stride=frame_stride)

    def test_step_collects_after_sufficient_frames(self):
        """Test that data is collected after sufficient frames are
        accumulated."""
        agent = FrameStackingAgent(num_frames=4, frame_stride=2)
        buffer: RandomReplacementBuffer[Tensor] = RandomReplacementBuffer(max_size=10)

        connected = connect_components(agent=agent, buffers={"video": buffer})
        data_user = connected.data_users["video"]
        agent.setup()

        # With num_frames=4 and frame_stride=2, needs 7 frames before collection
        for _ in range(6):
            agent.step(torch.randn(3, 64, 64))

        data_user.update()
        assert len(buffer) == 0

        # After 7th frame, should collect
        agent.step(torch.randn(3, 64, 64))
        data_user.update()
        assert len(buffer) == 1

        # Continues collecting on each step
        agent.step(torch.randn(3, 64, 64))
        data_user.update()
        assert len(buffer) == 2

        agent.teardown()

    def test_frame_stacking_with_stride(self):
        """Test that frames are stacked at correct stride intervals."""
        agent = FrameStackingAgent(num_frames=4, frame_stride=2)
        buffer: RandomReplacementBuffer[Tensor] = RandomReplacementBuffer(max_size=10)

        connected = connect_components(agent=agent, buffers={"video": buffer})
        data_user = connected.data_users["video"]
        agent.setup()

        # Feed 7 observations with unique values
        for i in range(7):
            obs = torch.full((1,), float(i))
            agent.step(obs)

        data_user.update()

        # Should collect frames [0, 2, 4, 6] as documented
        collected = buffer.get_data()[0]
        assert collected.shape == (4, 1)
        assert collected[0].item() == 0.0
        assert collected[1].item() == 2.0
        assert collected[2].item() == 4.0
        assert collected[3].item() == 6.0

        agent.teardown()

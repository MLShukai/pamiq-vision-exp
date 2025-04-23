import pytest
import torch
from pamiq_core import DataCollector
from pytest_mock import MockerFixture

from pamiq_vision_exp.agents.image_collecting import DataKeys, ImageCollectingAgent


class TestImageCollectingAgent:
    @pytest.fixture
    def mock_collector(self, mocker: MockerFixture):
        return mocker.MagicMock(DataCollector)

    @pytest.fixture
    def agent(self, mocker: MockerFixture, mock_collector):
        agent = ImageCollectingAgent()
        mock_get = mocker.patch.object(agent, "get_data_collector")
        mock_get.return_value = mock_collector
        agent.on_data_collectors_attached()
        return agent

    def test_step(self, agent: ImageCollectingAgent, mock_collector):
        assert agent.collector is mock_collector

        image = torch.randn(3, 64, 64)
        agent.step(image)

        mock_collector.collect.assert_called_once()
        arg = mock_collector.collect.call_args.args[0]
        assert isinstance(arg, dict)
        assert DataKeys.IMAGE in arg
        assert torch.equal(image, arg[DataKeys.IMAGE])

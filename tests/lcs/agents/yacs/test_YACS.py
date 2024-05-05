import pytest

from lcs import Perception
from lcs.agents.yacs.yacs import Configuration, YACS


class TestYACS:

    @pytest.fixture
    def cfg(self):
        return Configuration(
            classifier_length=2,
            number_of_possible_actions=2,
            feature_possible_values=[{'0', '1'}, {'0', '1'}])

    @pytest.fixture
    def agent(self, cfg):
        return YACS(cfg)

    def test_should_validate_if_perception_is_in_range(self, agent):
        with pytest.raises(AssertionError):
            agent.remember_situation(Perception('02'))

    def test_should_remember_perception(self, agent):
        # given
        assert len(agent.desirability_values) == 0

        # when & then
        p0 = Perception('00')
        agent.remember_situation(p0)
        assert len(agent.desirability_values) == 1
        assert p0 in agent.desirability_values

        p1 = Perception('01')
        agent.remember_situation(p1)
        assert len(agent.desirability_values) == 2
        assert p1 in agent.desirability_values

        p2 = Perception('00')
        agent.remember_situation(p2)
        assert len(agent.desirability_values) == 2

import pytest

from lcs import Perception
from lcs.agents.racs import Configuration, Effect
from lcs.representations import UBR
from lcs.representations.RealValueEncoder import RealValueEncoder


class TestEffect:

    @pytest.fixture
    def cfg(self):
        return Configuration(classifier_length=2,
                             number_of_possible_actions=2,
                             encoder=RealValueEncoder(4))

    @pytest.mark.parametrize("_e, _result", [
        ([UBR(0, 15), UBR(0, 15)], False),
        ([UBR(0, 15), UBR(2, 14)], True),
        ([UBR(4, 10), UBR(2, 14)], True),
    ])
    def test_should_detect_change(self, _e, _result, cfg):
        assert Effect(_e, cfg).specify_change == _result

    def test_should_create_pass_through_effect(self, cfg):
        # when
        effect = Effect.pass_through(cfg)

        # then
        assert len(effect) == cfg.classifier_length
        for allele in effect:
            assert allele == cfg.classifier_wildcard

    @pytest.mark.parametrize("_p0, _p1, _effect, is_specializable", [
        # Effect is all pass-through. Can be specialized.
        ([0.5, 0.5], [0.5, 0.5], [UBR(0, 15), UBR(0, 15)], True),
        # 1 pass-through effect get skipped. Second effect attribute get's
        # examined. P1 perception is not in correct range. That's invalid
        ([0.5, 0.5], [0.5, 0.5], [UBR(0, 15), UBR(2, 4)], False),
        # In this case the range is proper, but no change is anticipated.
        # In this case this should be a pass-through symbol.
        ([0.5, 0.5], [0.5, 0.5], [UBR(0, 15), UBR(2, 12)], False),
        # Here second perception attribute changes. 0.8 => 12
        ([0.5, 0.5], [0.5, 0.8], [UBR(0, 15), UBR(10, 14)], True)
    ])
    def test_should_specialize(self, _p0, _p1, _effect, is_specializable, cfg):
        # given
        p0 = Perception(_p0, oktypes=(float,))
        p1 = Perception(_p1, oktypes=(float,))
        effect = Effect(_effect, cfg=cfg)

        # then
        assert effect.is_specializable(p0, p1) is is_specializable

    @pytest.mark.parametrize("_effect1, _effect2, _result", [
        ([UBR(0, 15), UBR(0, 15)], [UBR(2, 4), UBR(5, 10)], True),
        ([UBR(6, 10), UBR(0, 15)], [UBR(2, 4), UBR(5, 10)], False),
        ([UBR(0, 15), UBR(4, 10)], [UBR(2, 4), UBR(6, 12)], False),
        ([UBR(2, 4), UBR(5, 5)], [UBR(2, 4), UBR(5, 5)], True),
    ])
    def test_should_subsume_effect(self, _effect1, _effect2, _result, cfg):
        # given
        effect1 = Effect(_effect1, cfg=cfg)
        effect2 = Effect(_effect2, cfg=cfg)

        # then
        assert effect1.subsumes(effect2) == _result

    @pytest.mark.parametrize("_effect, _result", [
        ([UBR(0, 15), UBR(0, 7)], 'OOOOOOOOOo|OOOOo.....')
    ])
    def test_should_visualize(self, _effect, _result, cfg):
        assert repr(Effect(_effect, cfg=cfg)) == _result

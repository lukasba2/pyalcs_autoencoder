from lcs.agents.xncs import Configuration


class TestEffect:
    def test_init(self):
        cfg = Configuration(2, 3, 4)
        assert cfg.lmc == 3
        assert cfg.lem == 4
        assert cfg.number_of_actions == 2


from nneve.quantum_oscillator.params import QOParams

ALT_C: float = 13.0
ALT_C_STEP: float = -3.0


class TestQOParams:
    def test_default_construct(self) -> None:
        params = QOParams()
        assert params.c == -2.0
        assert params.c_step == 0.16

    def test_manual_construct(self) -> None:
        params = QOParams(c=ALT_C, c_step=ALT_C_STEP)
        assert params.c == ALT_C
        assert params.c_step == ALT_C_STEP

    def test_run_update(self) -> None:
        params = QOParams()
        params.update()
        assert params.c == (-2.0 + 0.16)

    def test_alt_run_update(self) -> None:
        params = QOParams(c=ALT_C, c_step=ALT_C_STEP)
        params.update()
        assert params.c == (ALT_C + ALT_C_STEP)

    def test_get_extra(self) -> None:
        params = QOParams()
        extra = params.get_extra()
        assert isinstance(extra, tuple)
        assert len(extra) == 1
        assert extra[0] == -2.0

    def test_alt_get_extra(self) -> None:
        params = QOParams(c=ALT_C, c_step=ALT_C_STEP)
        extra = params.get_extra()
        assert isinstance(extra, tuple)
        assert len(extra) == 1
        assert extra[0] == ALT_C

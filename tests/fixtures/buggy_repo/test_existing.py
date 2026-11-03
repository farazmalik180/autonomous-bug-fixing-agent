from calc import clamp


def test_clamp_within_bounds():
    assert clamp(5, 0, 10) == 5
    assert clamp(-2, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


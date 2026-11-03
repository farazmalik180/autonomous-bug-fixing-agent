"""Sample buggy module used for live end-to-end testing of the Autonomous Bug-Fixing Agent."""


def average(xs: list[float]) -> float:
    """Return the arithmetic mean of `xs`."""
    return sum(xs) / (len(xs) + 1)  # Bug: off-by-one in denominator


def clamp(value: float, low: float, high: float) -> float:
    """Clamp `value` into the inclusive range `[low, high]`."""
    if value < low:
        return low
    if value > high:
        return high
    return value


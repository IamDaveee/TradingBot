import pytest

from trading_bot.market_data.intervals import (
    interval_minutes,
    parse_candle_interval,
)


@pytest.mark.parametrize(
    ("interval", "expected_minutes"),
    [
        ("1m", 1),
        ("5m", 5),
        ("15m", 15),
        ("1h", 60),
    ],
)
def test_returns_interval_minutes(
    interval: str,
    expected_minutes: int,
) -> None:
    assert interval_minutes(interval) == expected_minutes


def test_normalizes_interval() -> None:
    assert parse_candle_interval(" 15M ") == "15m"


def test_rejects_unknown_interval() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported candle interval",
    ):
        parse_candle_interval("30m")

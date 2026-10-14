from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandleInterval:
    name: str
    minutes: int


SUPPORTED_CANDLE_INTERVALS: dict[
    str,
    CandleInterval,
] = {
    "1m": CandleInterval(
        name="1m",
        minutes=1,
    ),
    "5m": CandleInterval(
        name="5m",
        minutes=5,
    ),
    "15m": CandleInterval(
        name="15m",
        minutes=15,
    ),
    "1h": CandleInterval(
        name="1h",
        minutes=60,
    ),
}


def parse_candle_interval(
    value: str,
) -> str:
    normalized = value.strip().lower()

    if normalized not in SUPPORTED_CANDLE_INTERVALS:
        supported = ", ".join(SUPPORTED_CANDLE_INTERVALS)

        raise ValueError(f"Unsupported candle interval: {value}. Supported intervals: {supported}")

    return normalized


def interval_minutes(
    interval: str,
) -> int:
    try:
        return SUPPORTED_CANDLE_INTERVALS[interval].minutes
    except KeyError as error:
        raise ValueError(f"Unsupported candle interval: {interval}") from error

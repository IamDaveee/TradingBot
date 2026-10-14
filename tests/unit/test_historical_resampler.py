from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)
from trading_bot.market_data.historical_resampler import (
    HistoricalCandleResampler,
)


def candle(
    minute: int,
    *,
    open_price: str | None = None,
    high_price: str | None = None,
    low_price: str | None = None,
    close_price: str | None = None,
    base_volume: str = "1",
    is_complete: bool = True,
) -> HistoricalCandle:
    start = datetime(
        2026,
        5,
        1,
        0,
        0,
        tzinfo=UTC,
    ) + timedelta(minutes=minute)

    default_price = Decimal("100") + Decimal(minute)

    open_value = Decimal(open_price) if open_price is not None else default_price

    high_value = Decimal(high_price) if high_price is not None else open_value + Decimal("2")

    low_value = Decimal(low_price) if low_price is not None else open_value - Decimal("2")

    close_value = Decimal(close_price) if close_price is not None else open_value + Decimal("1")

    volume = Decimal(base_volume)

    return HistoricalCandle(
        instrument_id="BTC-USDT",
        interval="1m",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=open_value,
        high_price=high_value,
        low_price=low_value,
        close_price=close_value,
        base_volume=volume,
        currency_volume=volume * Decimal("2"),
        quote_volume=volume * Decimal("3"),
        is_complete=is_complete,
    )


def test_resamples_complete_five_minute_bucket() -> None:
    source = [
        candle(
            0,
            open_price="100",
            high_price="105",
            low_price="99",
            close_price="101",
            base_volume="1",
        ),
        candle(
            1,
            open_price="101",
            high_price="106",
            low_price="100",
            close_price="102",
            base_volume="2",
        ),
        candle(
            2,
            open_price="102",
            high_price="108",
            low_price="98",
            close_price="103",
            base_volume="3",
        ),
        candle(
            3,
            open_price="103",
            high_price="107",
            low_price="101",
            close_price="104",
            base_volume="4",
        ),
        candle(
            4,
            open_price="104",
            high_price="109",
            low_price="102",
            close_price="105",
            base_volume="5",
        ),
    ]

    result = HistoricalCandleResampler().resample(
        source,
        target_interval="5m",
    )

    assert len(result.candles) == 1

    aggregated = result.candles[0]

    assert aggregated.interval == "5m"
    assert aggregated.open_price == Decimal("100")
    assert aggregated.high_price == Decimal("109")
    assert aggregated.low_price == Decimal("98")
    assert aggregated.close_price == Decimal("105")
    assert aggregated.base_volume == Decimal("15")
    assert aggregated.currency_volume == Decimal("30")
    assert aggregated.quote_volume == Decimal("45")

    assert aggregated.interval_start == datetime(
        2026,
        5,
        1,
        0,
        0,
        tzinfo=UTC,
    )

    assert aggregated.interval_end == datetime(
        2026,
        5,
        1,
        0,
        5,
        tzinfo=UTC,
    )


def test_skips_partial_bucket() -> None:
    source = [
        candle(0),
        candle(1),
        candle(2),
        candle(4),
    ]

    result = HistoricalCandleResampler().resample(
        source,
        target_interval="5m",
    )

    assert result.candles == []
    assert result.report.skipped_partial_bucket_count == 1


def test_skips_bucket_with_incomplete_candle() -> None:
    source = [
        candle(0),
        candle(1),
        candle(
            2,
            is_complete=False,
        ),
        candle(3),
        candle(4),
    ]

    result = HistoricalCandleResampler().resample(
        source,
        target_interval="5m",
    )

    assert result.candles == []
    assert result.report.skipped_incomplete_bucket_count == 1


def test_aligns_fifteen_minute_buckets() -> None:
    source = [candle(minute) for minute in range(30)]

    result = HistoricalCandleResampler().resample(
        source,
        target_interval="15m",
    )

    assert len(result.candles) == 2

    assert result.candles[0].interval_start == (
        datetime(
            2026,
            5,
            1,
            0,
            0,
            tzinfo=UTC,
        )
    )

    assert result.candles[1].interval_start == (
        datetime(
            2026,
            5,
            1,
            0,
            15,
            tzinfo=UTC,
        )
    )


def test_resamples_one_hour_bucket() -> None:
    source = [candle(minute) for minute in range(60)]

    result = HistoricalCandleResampler().resample(
        source,
        target_interval="1h",
    )

    assert len(result.candles) == 1
    assert result.candles[0].interval == "1h"
    assert result.candles[0].base_volume == (Decimal("60"))


def test_counts_duplicate_source_candles() -> None:
    source = [candle(minute) for minute in range(5)]

    source.append(
        candle(
            0,
            high_price="1000",
            close_price="999",
        )
    )

    result = HistoricalCandleResampler().resample(
        source,
        target_interval="5m",
    )

    assert len(result.candles) == 1

    assert result.report.duplicate_source_candle_count == 1

    assert result.candles[0].open_price == (Decimal("100"))


def test_rejects_unsupported_target_interval() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported target interval",
    ):
        HistoricalCandleResampler().resample(
            [candle(0)],
            target_interval="30m",
        )


def test_rejects_non_one_minute_source() -> None:
    source = candle(0)

    invalid = HistoricalCandle(
        instrument_id=source.instrument_id,
        interval="5m",
        interval_start=source.interval_start,
        interval_end=(source.interval_start + timedelta(minutes=5)),
        open_price=source.open_price,
        high_price=source.high_price,
        low_price=source.low_price,
        close_price=source.close_price,
        base_volume=source.base_volume,
        currency_volume=source.currency_volume,
        quote_volume=source.quote_volume,
        is_complete=True,
    )

    with pytest.raises(
        ValueError,
        match="All source candles",
    ):
        HistoricalCandleResampler().resample(
            [invalid],
            target_interval="5m",
        )

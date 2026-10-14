from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from trading_bot.market_data.candle_aggregator import (
    CandleAggregationError,
    TickerCandleAggregator,
)

SCHEMA = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "last_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "last_size",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "best_bid_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "best_ask_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "spread",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "exchange_timestamp",
            pa.timestamp(
                "us",
                tz="UTC",
            ),
            nullable=False,
        ),
        pa.field(
            "received_timestamp",
            pa.timestamp(
                "us",
                tz="UTC",
            ),
            nullable=False,
        ),
    ]
)


def create_record(
    price: str,
    size: str,
    bid: str,
    ask: str,
    timestamp: str,
    symbol: str = "BTC-USDT",
) -> dict[str, object]:
    parsed_timestamp = datetime.fromisoformat(timestamp).astimezone(UTC)

    return {
        "instrument_id": symbol,
        "last_price": Decimal(price),
        "last_size": Decimal(size),
        "best_bid_price": Decimal(bid),
        "best_ask_price": Decimal(ask),
        "spread": (Decimal(ask) - Decimal(bid)),
        "exchange_timestamp": (parsed_timestamp),
        "received_timestamp": (parsed_timestamp),
    }


def create_table(
    records: list[dict[str, object]],
) -> pa.Table:
    return pa.Table.from_pylist(
        records,
        schema=SCHEMA,
    )


def test_aggregates_one_minute_candle() -> None:
    table = create_table(
        [
            create_record(
                price="100",
                size="1",
                bid="99.9",
                ask="100.1",
                timestamp=("2026-06-27T12:00:10+00:00"),
            ),
            create_record(
                price="103",
                size="2",
                bid="102.9",
                ask="103.1",
                timestamp=("2026-06-27T12:00:20+00:00"),
            ),
            create_record(
                price="98",
                size="3",
                bid="97.8",
                ask="98.2",
                timestamp=("2026-06-27T12:00:30+00:00"),
            ),
            create_record(
                price="101",
                size="4",
                bid="100.9",
                ask="101.1",
                timestamp=("2026-06-27T12:00:50+00:00"),
            ),
        ]
    )

    aggregator = TickerCandleAggregator()

    candles = aggregator.aggregate(table)

    assert len(candles) == 1

    candle = candles[0]

    assert candle.instrument_id == "BTC-USDT"
    assert candle.open_price == Decimal("100")
    assert candle.high_price == Decimal("103")
    assert candle.low_price == Decimal("98")
    assert candle.close_price == Decimal("101")
    assert candle.observed_size == Decimal("10")
    assert candle.tick_count == 4

    assert candle.interval_start == datetime(
        2026,
        6,
        27,
        12,
        0,
        tzinfo=UTC,
    )

    assert candle.interval_end == datetime(
        2026,
        6,
        27,
        12,
        1,
        tzinfo=UTC,
    )


def test_separates_minute_intervals() -> None:
    table = create_table(
        [
            create_record(
                price="100",
                size="1",
                bid="99.9",
                ask="100.1",
                timestamp=("2026-06-27T12:00:59+00:00"),
            ),
            create_record(
                price="101",
                size="1",
                bid="100.9",
                ask="101.1",
                timestamp=("2026-06-27T12:01:00+00:00"),
            ),
        ]
    )

    candles = TickerCandleAggregator().aggregate(table)

    assert len(candles) == 2

    assert candles[0].interval_start.minute == 0
    assert candles[1].interval_start.minute == 1


def test_separates_symbols() -> None:
    table = create_table(
        [
            create_record(
                price="100",
                size="1",
                bid="99.9",
                ask="100.1",
                timestamp=("2026-06-27T12:00:10+00:00"),
                symbol="BTC-USDT",
            ),
            create_record(
                price="200",
                size="1",
                bid="199.9",
                ask="200.1",
                timestamp=("2026-06-27T12:00:10+00:00"),
                symbol="ETH-USDT",
            ),
        ]
    )

    candles = TickerCandleAggregator().aggregate(table)

    assert len(candles) == 2

    assert {candle.instrument_id for candle in candles} == {
        "BTC-USDT",
        "ETH-USDT",
    }


def test_sorts_unsorted_records() -> None:
    table = create_table(
        [
            create_record(
                price="105",
                size="1",
                bid="104.9",
                ask="105.1",
                timestamp=("2026-06-27T12:00:50+00:00"),
            ),
            create_record(
                price="100",
                size="1",
                bid="99.9",
                ask="100.1",
                timestamp=("2026-06-27T12:00:10+00:00"),
            ),
        ]
    )

    candle = TickerCandleAggregator().aggregate(table)[0]

    assert candle.open_price == Decimal("100")
    assert candle.close_price == Decimal("105")


def test_rejects_crossed_market() -> None:
    table = create_table(
        [
            create_record(
                price="100",
                size="1",
                bid="100.2",
                ask="100.1",
                timestamp=("2026-06-27T12:00:10+00:00"),
            )
        ]
    )

    with pytest.raises(CandleAggregationError):
        TickerCandleAggregator().aggregate(table)


def test_empty_table_returns_no_candles() -> None:
    table = pa.Table.from_pylist(
        [],
        schema=SCHEMA,
    )

    candles = TickerCandleAggregator().aggregate(table)

    assert candles == []


def test_rejects_invalid_interval() -> None:
    table = pa.Table.from_pylist(
        [],
        schema=SCHEMA,
    )

    with pytest.raises(ValueError):
        TickerCandleAggregator().aggregate(
            table,
            interval_minutes=0,
        )

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pyarrow as pa

from trading_bot.validation.candle_validator import (
    CandleValidator,
)


def create_record(
    minute: int,
) -> dict[str, object]:
    start = datetime(
        2026,
        6,
        27,
        12,
        minute,
        tzinfo=UTC,
    )

    return {
        "instrument_id": "BTC-USDT",
        "interval_start": start,
        "interval_end": (start + timedelta(minutes=1)),
        "open_price": Decimal("100"),
        "high_price": Decimal("102"),
        "low_price": Decimal("99"),
        "close_price": Decimal("101"),
        "minimum_spread": Decimal("0.1"),
        "maximum_spread": Decimal("0.3"),
        "average_spread": Decimal("0.2"),
        "observed_size": Decimal("10"),
        "tick_count": 5,
        "first_exchange_timestamp": start,
        "last_exchange_timestamp": (start + timedelta(seconds=50)),
        "first_received_timestamp": start,
        "last_received_timestamp": (start + timedelta(seconds=50)),
    }


def create_table(
    records: list[dict[str, object]],
) -> pa.Table:
    return pa.Table.from_pylist(records)


def test_valid_candles_pass() -> None:
    table = create_table(
        [
            create_record(0),
            create_record(1),
        ]
    )

    report = CandleValidator().validate(table)

    assert report.is_valid
    assert report.invalid_records == 0
    assert report.duplicate_intervals == 0
    assert report.missing_intervals == 0


def test_detects_duplicate_interval() -> None:
    table = create_table(
        [
            create_record(0),
            create_record(0),
        ]
    )

    report = CandleValidator().validate(table)

    assert not report.is_valid
    assert report.duplicate_intervals == 1


def test_detects_missing_interval() -> None:
    table = create_table(
        [
            create_record(0),
            create_record(2),
        ]
    )

    report = CandleValidator().validate(table)

    assert report.missing_intervals == 1


def test_detects_invalid_ohlc() -> None:
    record = create_record(0)
    record["high_price"] = Decimal("98")

    table = create_table([record])

    report = CandleValidator().validate(table)

    assert not report.is_valid
    assert report.invalid_records == 1


def test_large_gap_is_capture_session_gap() -> None:
    table = create_table(
        [
            create_record(0),
            {
                **create_record(1),
                "interval_start": datetime(
                    2026,
                    6,
                    27,
                    14,
                    0,
                    tzinfo=UTC,
                ),
                "interval_end": datetime(
                    2026,
                    6,
                    27,
                    14,
                    1,
                    tzinfo=UTC,
                ),
                "first_exchange_timestamp": datetime(
                    2026,
                    6,
                    27,
                    14,
                    0,
                    tzinfo=UTC,
                ),
                "last_exchange_timestamp": datetime(
                    2026,
                    6,
                    27,
                    14,
                    0,
                    50,
                    tzinfo=UTC,
                ),
                "first_received_timestamp": datetime(
                    2026,
                    6,
                    27,
                    14,
                    0,
                    tzinfo=UTC,
                ),
                "last_received_timestamp": datetime(
                    2026,
                    6,
                    27,
                    14,
                    0,
                    50,
                    tzinfo=UTC,
                ),
            },
        ]
    )

    report = CandleValidator().validate(table)

    assert report.missing_intervals == 0
    assert report.session_gaps == 1
    assert report.is_valid

from datetime import UTC, datetime
from decimal import Decimal

from trading_bot.exchanges.okx.models import OkxTicker

received_timestamp = datetime(
    2026,
    6,
    26,
    12,
    0,
    1,
    tzinfo=UTC,
)


def test_okx_ticker_is_parsed_from_payload() -> None:
    payload: dict[str, object] = {
        "instId": "BTC-USDT",
        "last": "107500.1",
        "lastSz": "0.002",
        "askPx": "107500.2",
        "askSz": "1.25",
        "bidPx": "107500.1",
        "bidSz": "0.98",
        "ts": "1782475200000",
    }

    ticker = OkxTicker.from_okx_payload(
        payload,
        received_timestamp=received_timestamp,
    )

    assert ticker.instrument_id == "BTC-USDT"
    assert ticker.last_price == Decimal("107500.1")
    assert ticker.best_ask_price == Decimal("107500.2")
    assert ticker.best_bid_price == Decimal("107500.1")
    assert ticker.spread == Decimal("0.1")
    assert ticker.mid_price == Decimal("107500.15")
    assert ticker.received_timestamp == received_timestamp
    assert ticker.data_age_seconds == 1.0
    assert ticker.clock_offset_detected is False
    assert ticker.exchange_timestamp == datetime(
        2026,
        6,
        26,
        12,
        0,
        tzinfo=UTC,
    )


def test_okx_ticker_calculates_spread_percentage() -> None:
    ticker = OkxTicker(
        instrument_id="BTC-USDT",
        last_price=Decimal("100.05"),
        last_size=Decimal("1"),
        best_ask_price=Decimal("100.10"),
        best_ask_size=Decimal("2"),
        best_bid_price=Decimal("100.00"),
        best_bid_size=Decimal("3"),
        exchange_timestamp=datetime.now(UTC),
        received_timestamp=datetime.now(UTC),
    )

    assert ticker.spread == Decimal("0.10")
    assert ticker.mid_price == Decimal("100.05")
    assert ticker.spread_percentage > Decimal("0")


def test_okx_ticker_detects_negative_data_age() -> None:
    exchange_timestamp = datetime(
        2026,
        6,
        26,
        12,
        0,
        1,
        tzinfo=UTC,
    )

    received_timestamp = datetime(
        2026,
        6,
        26,
        12,
        0,
        0,
        tzinfo=UTC,
    )

    ticker = OkxTicker(
        instrument_id="BTC-USDT",
        last_price=Decimal("100"),
        last_size=Decimal("1"),
        best_ask_price=Decimal("100.1"),
        best_ask_size=Decimal("1"),
        best_bid_price=Decimal("99.9"),
        best_bid_size=Decimal("1"),
        exchange_timestamp=exchange_timestamp,
        received_timestamp=received_timestamp,
    )

    assert ticker.data_age_seconds == -1.0
    assert ticker.clock_offset_detected is True

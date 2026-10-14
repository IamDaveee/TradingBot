from decimal import Decimal

import pytest
from pydantic import ValidationError

from trading_bot.exchanges.okx.models import OkxInstrument


def test_instrument_is_parsed_correctly() -> None:
    instrument = OkxInstrument(
        instrument_id="BTC-USDT",
        instrument_type="SPOT",
        base_currency="BTC",
        quote_currency="USDT",
        state="live",
        tick_size=Decimal("0.1"),
        lot_size=Decimal("0.00000001"),
        minimum_order_size=Decimal("0.00001"),
    )

    assert instrument.instrument_id == "BTC-USDT"
    assert instrument.tick_size == Decimal("0.1")
    assert instrument.lot_size == Decimal("0.00000001")
    assert instrument.minimum_order_size == Decimal("0.00001")
    assert instrument.is_live is True


def test_instrument_rejects_zero_tick_size() -> None:
    with pytest.raises(ValidationError):
        OkxInstrument(
            instrument_id="BTC-USDT",
            instrument_type="SPOT",
            base_currency="BTC",
            quote_currency="USDT",
            state="live",
            tick_size=Decimal("0"),
            lot_size=Decimal("0.00000001"),
            minimum_order_size=Decimal("0.00001"),
        )

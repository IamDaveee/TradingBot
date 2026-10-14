from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.sweep import (
    SweepConfiguration,
)
from trading_bot.backtesting.walk_forward import (
    WalkForwardSettings,
    WalkForwardValidator,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


def candle(
    minute: int,
    price: str,
) -> TickerCandle:
    start = datetime(
        2026,
        5,
        1,
        0,
        0,
        tzinfo=UTC,
    ) + timedelta(minutes=minute)

    value = Decimal(price)

    return TickerCandle(
        instrument_id="BTC-USDT",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=value,
        high_price=value,
        low_price=value,
        close_price=value,
        first_bid_price=(value - Decimal("0.1")),
        last_bid_price=(value - Decimal("0.1")),
        first_ask_price=(value + Decimal("0.1")),
        last_ask_price=(value + Decimal("0.1")),
        minimum_spread=Decimal("0.2"),
        maximum_spread=Decimal("0.2"),
        average_spread=Decimal("0.2"),
        observed_size=Decimal("1"),
        tick_count=1,
        first_exchange_timestamp=start,
        last_exchange_timestamp=start,
        first_received_timestamp=start,
        last_received_timestamp=start,
    )


def create_multi_day_candles(
    days: int,
) -> list[TickerCandle]:
    candles: list[TickerCandle] = []

    total_minutes = days * 24 * 60

    for minute in range(total_minutes):
        cycle = minute % 120

        if cycle < 60:
            price = Decimal("100") + Decimal(cycle)
        else:
            price = Decimal("160") - Decimal(cycle - 60)

        candles.append(
            candle(
                minute,
                str(price),
            )
        )

    return candles


def test_settings_reject_overlapping_test_windows() -> None:
    with pytest.raises(ValueError):
        WalkForwardSettings(
            training_days=3,
            testing_days=2,
            step_days=1,
        )


def test_creates_expected_fold_count() -> None:
    candles = create_multi_day_candles(days=8)

    validator = WalkForwardValidator(
        settings=WalkForwardSettings(
            training_days=3,
            testing_days=2,
            step_days=2,
        ),
        initial_cash=Decimal("1000"),
        allocation_fraction=Decimal("0.5"),
        fee_rate=Decimal("0"),
        slippage_basis_points=Decimal("0"),
    )

    result = validator.run(
        candles=candles,
        configurations=[
            SweepConfiguration(
                fast_window=2,
                slow_window=5,
            ),
            SweepConfiguration(
                fast_window=3,
                slow_window=8,
            ),
        ],
    )

    assert result.summary.fold_count == 2
    assert len(result.folds) == 2

    assert result.folds[0].training_start.isoformat() == ("2026-05-01")

    assert result.folds[0].training_end.isoformat() == ("2026-05-03")

    assert result.folds[0].testing_start.isoformat() == ("2026-05-04")

    assert result.folds[0].testing_end.isoformat() == ("2026-05-05")


def test_carries_equity_between_folds() -> None:
    candles = create_multi_day_candles(days=8)

    validator = WalkForwardValidator(
        settings=WalkForwardSettings(
            training_days=3,
            testing_days=2,
            step_days=2,
        ),
        initial_cash=Decimal("1000"),
        allocation_fraction=Decimal("0.5"),
        fee_rate=Decimal("0"),
        slippage_basis_points=Decimal("0"),
    )

    result = validator.run(
        candles=candles,
        configurations=[
            SweepConfiguration(
                fast_window=2,
                slow_window=5,
            )
        ],
    )

    assert result.folds[1].testing_initial_equity == result.folds[0].testing_final_equity

    assert result.summary.final_equity == result.folds[-1].testing_final_equity


def test_rejects_multiple_symbols() -> None:
    candles = create_multi_day_candles(days=6)

    first = candles[0]

    candles.append(
        TickerCandle(
            instrument_id="ETH-USDT",
            interval_start=(first.interval_start),
            interval_end=first.interval_end,
            open_price=first.open_price,
            high_price=first.high_price,
            low_price=first.low_price,
            close_price=first.close_price,
            first_bid_price=(first.first_bid_price),
            last_bid_price=(first.last_bid_price),
            first_ask_price=(first.first_ask_price),
            last_ask_price=(first.last_ask_price),
            minimum_spread=(first.minimum_spread),
            maximum_spread=(first.maximum_spread),
            average_spread=(first.average_spread),
            observed_size=(first.observed_size),
            tick_count=first.tick_count,
            first_exchange_timestamp=(first.first_exchange_timestamp),
            last_exchange_timestamp=(first.last_exchange_timestamp),
            first_received_timestamp=(first.first_received_timestamp),
            last_received_timestamp=(first.last_received_timestamp),
        )
    )

    validator = WalkForwardValidator(
        settings=WalkForwardSettings(
            training_days=2,
            testing_days=1,
            step_days=1,
        ),
        initial_cash=Decimal("1000"),
        allocation_fraction=Decimal("0.5"),
        fee_rate=Decimal("0"),
        slippage_basis_points=Decimal("0"),
    )

    with pytest.raises(
        ValueError,
        match="exactly one instrument",
    ):
        validator.run(
            candles=candles,
            configurations=[
                SweepConfiguration(
                    fast_window=2,
                    slow_window=5,
                )
            ],
        )


def test_rejects_dataset_too_short() -> None:
    candles = create_multi_day_candles(days=3)

    validator = WalkForwardValidator(
        settings=WalkForwardSettings(
            training_days=3,
            testing_days=2,
            step_days=2,
        ),
        initial_cash=Decimal("1000"),
        allocation_fraction=Decimal("0.5"),
        fee_rate=Decimal("0"),
        slippage_basis_points=Decimal("0"),
    )

    with pytest.raises(
        ValueError,
        match="too short",
    ):
        validator.run(
            candles=candles,
            configurations=[
                SweepConfiguration(
                    fast_window=2,
                    slow_window=5,
                )
            ],
        )

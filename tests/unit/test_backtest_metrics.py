from datetime import UTC, datetime
from decimal import Decimal

from trading_bot.backtesting.execution import (
    BacktestResult,
    EquitySnapshot,
)
from trading_bot.backtesting.metrics import (
    calculate_backtest_metrics,
)


def snapshot(
    minute: int,
    equity: str,
    fees: str = "0",
) -> EquitySnapshot:
    return EquitySnapshot(
        timestamp=datetime(
            2026,
            6,
            27,
            12,
            minute,
            tzinfo=UTC,
        ),
        cash=Decimal(equity),
        equity=Decimal(equity),
        realized_pnl=Decimal("0"),
        fees_paid=Decimal(fees),
    )


def test_calculates_return() -> None:
    result = BacktestResult(
        equity_curve=[
            snapshot(0, "1000"),
            snapshot(1, "1100"),
        ]
    )

    metrics = calculate_backtest_metrics(
        result=result,
        initial_equity=Decimal("1000"),
    )

    assert metrics.absolute_return == Decimal("100")

    assert metrics.return_percentage == Decimal("10.0")


def test_calculates_maximum_drawdown() -> None:
    result = BacktestResult(
        equity_curve=[
            snapshot(0, "1000"),
            snapshot(1, "1200"),
            snapshot(2, "900"),
            snapshot(3, "1100"),
        ]
    )

    metrics = calculate_backtest_metrics(
        result=result,
        initial_equity=Decimal("1000"),
    )

    assert metrics.maximum_drawdown == Decimal("300")

    assert metrics.maximum_drawdown_percentage == Decimal("25.00")

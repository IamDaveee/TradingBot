from dataclasses import dataclass
from decimal import Decimal

from trading_bot.backtesting.execution import (
    BacktestResult,
)


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    initial_equity: Decimal
    final_equity: Decimal
    absolute_return: Decimal
    return_percentage: Decimal
    maximum_drawdown: Decimal
    maximum_drawdown_percentage: Decimal
    fill_count: int
    rejected_order_count: int
    unexecuted_order_count: int
    total_fees: Decimal


def calculate_backtest_metrics(
    result: BacktestResult,
    initial_equity: Decimal,
) -> BacktestMetrics:
    if not initial_equity.is_finite():
        raise ValueError("initial_equity must be finite")

    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than zero")

    final_equity = result.final_equity if result.final_equity is not None else initial_equity

    absolute_return = final_equity - initial_equity

    return_percentage = absolute_return / initial_equity * Decimal("100")

    (
        maximum_drawdown,
        maximum_drawdown_percentage,
    ) = _calculate_maximum_drawdown(result)

    total_fees = result.equity_curve[-1].fees_paid if result.equity_curve else Decimal("0")

    return BacktestMetrics(
        initial_equity=initial_equity,
        final_equity=final_equity,
        absolute_return=absolute_return,
        return_percentage=return_percentage,
        maximum_drawdown=maximum_drawdown,
        maximum_drawdown_percentage=(maximum_drawdown_percentage),
        fill_count=len(result.fills),
        rejected_order_count=len(result.rejected_orders),
        unexecuted_order_count=len(result.unexecuted_orders),
        total_fees=total_fees,
    )


def _calculate_maximum_drawdown(
    result: BacktestResult,
) -> tuple[Decimal, Decimal]:
    if not result.equity_curve:
        return (
            Decimal("0"),
            Decimal("0"),
        )

    peak = result.equity_curve[0].equity
    maximum_drawdown = Decimal("0")
    maximum_percentage = Decimal("0")

    for snapshot in result.equity_curve:
        equity = snapshot.equity

        if equity > peak:
            peak = equity
            continue

        drawdown = peak - equity

        if drawdown > maximum_drawdown:
            maximum_drawdown = drawdown

        if peak > 0:
            percentage = drawdown / peak * Decimal("100")

            if percentage > maximum_percentage:
                maximum_percentage = percentage

    return (
        maximum_drawdown,
        maximum_percentage,
    )

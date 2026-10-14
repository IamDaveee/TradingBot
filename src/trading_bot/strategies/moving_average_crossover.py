from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from trading_bot.backtesting.models import (
    OrderRequest,
    OrderSide,
)
from trading_bot.backtesting.strategy import (
    StrategyContext,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


@dataclass(slots=True)
class MovingAverageCrossoverStrategy:
    fast_window: int
    slow_window: int
    allocation_fraction: Decimal
    _close_prices: dict[
        str,
        deque[Decimal],
    ] = field(
        default_factory=dict,
        init=False,
    )
    _previous_relation: dict[
        str,
        int,
    ] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.fast_window <= 0:
            raise ValueError("fast_window must be greater than zero")

        if self.slow_window <= 0:
            raise ValueError("slow_window must be greater than zero")

        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be below slow_window")

        if not self.allocation_fraction.is_finite():
            raise ValueError("allocation_fraction must be finite")

        if not (Decimal("0") < self.allocation_fraction <= Decimal("1")):
            raise ValueError("allocation_fraction must be within (0, 1]")

    def on_candle(
        self,
        candle: TickerCandle,
        context: StrategyContext,
    ) -> list[OrderRequest]:
        prices = self._close_prices.setdefault(
            candle.instrument_id,
            deque(
                maxlen=self.slow_window,
            ),
        )

        prices.append(candle.close_price)

        if len(prices) < self.slow_window:
            return []

        price_list = list(prices)

        fast_average = sum(
            price_list[-self.fast_window :],
            start=Decimal("0"),
        ) / Decimal(self.fast_window)

        slow_average = sum(
            price_list,
            start=Decimal("0"),
        ) / Decimal(self.slow_window)

        current_relation = self._compare(
            fast_average,
            slow_average,
        )

        previous_relation = self._previous_relation.get(candle.instrument_id)

        self._previous_relation[candle.instrument_id] = current_relation

        if previous_relation is None:
            return []

        crossed_above = previous_relation <= 0 and current_relation > 0

        crossed_below = previous_relation >= 0 and current_relation < 0

        if crossed_above and not context.position.is_open:
            quantity = self._calculate_buy_quantity(
                cash=context.cash,
                reference_price=candle.close_price,
            )

            if quantity <= 0:
                return []

            return [
                OrderRequest(
                    instrument_id=(candle.instrument_id),
                    side=OrderSide.BUY,
                    quantity=quantity,
                    created_at=(candle.interval_end),
                )
            ]

        if crossed_below and context.position.is_open:
            return [
                OrderRequest(
                    instrument_id=(candle.instrument_id),
                    side=OrderSide.SELL,
                    quantity=(context.position.quantity),
                    created_at=(candle.interval_end),
                )
            ]

        return []

    def _calculate_buy_quantity(
        self,
        cash: Decimal,
        reference_price: Decimal,
    ) -> Decimal:
        if cash <= 0:
            return Decimal("0")

        allocation = cash * self.allocation_fraction

        return allocation / reference_price

    @staticmethod
    def _compare(
        left: Decimal,
        right: Decimal,
    ) -> int:
        if left > right:
            return 1

        if left < right:
            return -1

        return 0

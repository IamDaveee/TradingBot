from collections import deque
from decimal import Decimal

from trading_bot.backtesting.models import (
    OrderRequest,
    OrderSide,
    OrderType,
)
from trading_bot.backtesting.strategy import StrategyContext
from trading_bot.indicators.atr import AverageTrueRange
from trading_bot.market_data.candle import TickerCandle


class AtrFilteredMovingAverageCrossoverStrategy:
    def __init__(
        self,
        fast_window: int,
        slow_window: int,
        atr_period: int,
        atr_multiplier: Decimal,
        allocation_fraction: Decimal,
        estimated_round_trip_cost_rate: Decimal,
        minimum_holding_candles: int = 0,
        cooldown_candles: int = 0,
    ) -> None:
        if fast_window <= 0:
            raise ValueError("fast_window must be positive")

        if slow_window <= fast_window:
            raise ValueError("slow_window must be greater than fast_window")

        if atr_period <= 0:
            raise ValueError("atr_period must be positive")

        if not atr_multiplier.is_finite() or atr_multiplier < 0:
            raise ValueError("atr_multiplier must be a non-negative finite Decimal")

        if (
            not allocation_fraction.is_finite()
            or allocation_fraction <= 0
            or allocation_fraction > 1
        ):
            raise ValueError("allocation_fraction must be within (0, 1]")

        if (
            not estimated_round_trip_cost_rate.is_finite()
            or estimated_round_trip_cost_rate < 0
            or estimated_round_trip_cost_rate >= 1
        ):
            raise ValueError("estimated_round_trip_cost_rate must be within [0, 1)")

        if minimum_holding_candles < 0:
            raise ValueError("minimum_holding_candles must not be negative")

        if cooldown_candles < 0:
            raise ValueError("cooldown_candles must not be negative")

        self._fast_window = fast_window
        self._slow_window = slow_window
        self._atr_multiplier = atr_multiplier
        self._allocation_fraction = allocation_fraction

        self._estimated_round_trip_cost_rate = estimated_round_trip_cost_rate

        self._minimum_holding_candles = minimum_holding_candles

        self._cooldown_candles = cooldown_candles

        self._closes: deque[Decimal] = deque(maxlen=slow_window)

        self._atr = AverageTrueRange(period=atr_period)

        self._previous_fast: Decimal | None = None
        self._previous_slow: Decimal | None = None

        self._holding_candles = 0
        self._remaining_cooldown = 0

    def on_candle(
        self,
        candle: TickerCandle,
        context: StrategyContext,
    ) -> list[OrderRequest]:
        self._closes.append(candle.close_price)

        atr_value = self._atr.update(candle)

        position_quantity = context.position.quantity

        has_position = position_quantity > 0

        if has_position:
            self._holding_candles += 1
        else:
            self._holding_candles = 0

            if self._remaining_cooldown > 0:
                self._remaining_cooldown -= 1

        if len(self._closes) < self._slow_window or atr_value is None:
            return []

        closes = list(self._closes)

        fast_average = sum(
            closes[-self._fast_window :],
            start=Decimal("0"),
        ) / Decimal(self._fast_window)

        slow_average = sum(
            closes,
            start=Decimal("0"),
        ) / Decimal(self._slow_window)

        if self._previous_fast is None or self._previous_slow is None:
            self._previous_fast = fast_average
            self._previous_slow = slow_average
            return []

        bearish_crossover = (
            self._previous_fast >= self._previous_slow and fast_average < slow_average
        )

        self._previous_fast = fast_average
        self._previous_slow = slow_average

        if (
            has_position
            and bearish_crossover
            and self._holding_candles >= self._minimum_holding_candles
        ):
            self._remaining_cooldown = self._cooldown_candles

            return [
                OrderRequest(
                    instrument_id=(candle.instrument_id),
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position_quantity,
                    created_at=candle.interval_end,
                )
            ]

        if has_position:
            return []

        if self._remaining_cooldown > 0:
            return []

        bullish_regime = fast_average > slow_average

        if not bullish_regime:
            return []

        ma_separation = fast_average - slow_average

        atr_threshold = atr_value * self._atr_multiplier

        cost_threshold = candle.close_price * self._estimated_round_trip_cost_rate

        required_separation = max(
            atr_threshold,
            cost_threshold,
        )

        if ma_separation < required_separation:
            return []

        allocation_value = context.cash * self._allocation_fraction

        if allocation_value <= 0:
            return []

        quantity = allocation_value / candle.close_price

        if quantity <= 0:
            return []

        return [
            OrderRequest(
                instrument_id=candle.instrument_id,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                created_at=candle.interval_end,
            )
        ]

from collections import deque
from decimal import Decimal

from trading_bot.market_data.candle import TickerCandle


class AverageTrueRange:
    """
    Simple moving-average ATR.

    The first candle contributes high - low because no previous
    close is available yet.
    """

    def __init__(
        self,
        period: int,
    ) -> None:
        if period <= 0:
            raise ValueError("period must be greater than zero")

        self._period = period
        self._previous_close: Decimal | None = None

        self._true_ranges: deque[Decimal] = deque(maxlen=period)

    @property
    def period(self) -> int:
        return self._period

    @property
    def is_ready(self) -> bool:
        return len(self._true_ranges) == self._period

    @property
    def value(self) -> Decimal | None:
        if not self.is_ready:
            return None

        return sum(
            self._true_ranges,
            start=Decimal("0"),
        ) / Decimal(self._period)

    def update(
        self,
        candle: TickerCandle,
    ) -> Decimal | None:
        high_low = candle.high_price - candle.low_price

        if self._previous_close is None:
            true_range = high_low
        else:
            high_previous = abs(candle.high_price - self._previous_close)

            low_previous = abs(candle.low_price - self._previous_close)

            true_range = max(
                high_low,
                high_previous,
                low_previous,
            )

        self._true_ranges.append(true_range)

        self._previous_close = candle.close_price

        return self.value

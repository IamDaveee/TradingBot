from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class HistoricalCandle:
    instrument_id: str
    interval: str
    interval_start: datetime
    interval_end: datetime

    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal

    base_volume: Decimal
    currency_volume: Decimal
    quote_volume: Decimal

    is_complete: bool

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")

        if not self.interval:
            raise ValueError("interval must not be empty")

        if self.interval_start.tzinfo is None:
            raise ValueError("interval_start must contain a timezone")

        if self.interval_end.tzinfo is None:
            raise ValueError("interval_end must contain a timezone")

        if self.interval_end <= self.interval_start:
            raise ValueError("interval_end must be after interval_start")

        decimal_values = {
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "base_volume": self.base_volume,
            "currency_volume": self.currency_volume,
            "quote_volume": self.quote_volume,
        }

        for name, value in decimal_values.items():
            if not value.is_finite():
                raise ValueError(f"{name} must be finite")

        price_values = {
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
        }

        for name, value in price_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        volume_values = {
            "base_volume": self.base_volume,
            "currency_volume": self.currency_volume,
            "quote_volume": self.quote_volume,
        }

        for name, value in volume_values.items():
            if value < 0:
                raise ValueError(f"{name} must not be negative")

        if self.high_price < self.low_price:
            raise ValueError("high_price must not be below low_price")

        if not (self.low_price <= self.open_price <= self.high_price):
            raise ValueError("open_price must be within the candle range")

        if not (self.low_price <= self.close_price <= self.high_price):
            raise ValueError("close_price must be within the candle range")

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class TickerCandle:
    instrument_id: str

    interval_start: datetime
    interval_end: datetime

    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal

    first_bid_price: Decimal
    last_bid_price: Decimal
    first_ask_price: Decimal
    last_ask_price: Decimal

    minimum_spread: Decimal
    maximum_spread: Decimal
    average_spread: Decimal

    observed_size: Decimal
    tick_count: int

    first_exchange_timestamp: datetime
    last_exchange_timestamp: datetime
    first_received_timestamp: datetime
    last_received_timestamp: datetime

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")

        if self.interval_start.tzinfo is None:
            raise ValueError("interval_start must contain a timezone")

        if self.interval_end.tzinfo is None:
            raise ValueError("interval_end must contain a timezone")

        if self.interval_end <= self.interval_start:
            raise ValueError("interval_end must be after interval_start")

        if self.tick_count <= 0:
            raise ValueError("tick_count must be greater than zero")

        if self.observed_size < 0:
            raise ValueError("observed_size must not be negative")

        if self.high_price < self.low_price:
            raise ValueError("high_price must not be below low_price")

        if not (self.low_price <= self.open_price <= self.high_price):
            raise ValueError("open_price must be within candle range")

        if not (self.low_price <= self.close_price <= self.high_price):
            raise ValueError("close_price must be within candle range")

        if self.minimum_spread < 0:
            raise ValueError("minimum_spread must not be negative")

        if self.maximum_spread < self.minimum_spread:
            raise ValueError("maximum_spread must not be below minimum_spread")

        if not (self.minimum_spread <= self.average_spread <= self.maximum_spread):
            raise ValueError("average_spread must be within the spread range")

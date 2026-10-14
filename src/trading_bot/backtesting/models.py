from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    instrument_id: str
    side: OrderSide
    quantity: Decimal
    created_at: datetime
    order_type: OrderType = OrderType.MARKET

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero")

        if not self.quantity.is_finite():
            raise ValueError("quantity must be finite")

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must contain a timezone")


@dataclass(frozen=True, slots=True)
class Fill:
    instrument_id: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero")

        if self.price <= 0:
            raise ValueError("price must be greater than zero")

        if self.fee < 0:
            raise ValueError("fee must not be negative")

        decimal_values = {
            "quantity": self.quantity,
            "price": self.price,
            "fee": self.fee,
        }

        for name, value in decimal_values.items():
            if not value.is_finite():
                raise ValueError(f"{name} must be finite")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must contain a timezone")

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price

    @property
    def cash_change(self) -> Decimal:
        if self.side == OrderSide.BUY:
            return -(self.notional + self.fee)

        return self.notional - self.fee

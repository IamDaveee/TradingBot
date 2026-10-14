from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PercentageFeeModel:
    rate: Decimal

    def __post_init__(self) -> None:
        if not self.rate.is_finite():
            raise ValueError("Fee rate must be finite")

        if self.rate < 0:
            raise ValueError("Fee rate must not be negative")

        if self.rate >= 1:
            raise ValueError("Fee rate must be below 1")

    def calculate(
        self,
        quantity: Decimal,
        price: Decimal,
    ) -> Decimal:
        if quantity <= 0:
            raise ValueError("quantity must be greater than zero")

        if price <= 0:
            raise ValueError("price must be greater than zero")

        return quantity * price * self.rate

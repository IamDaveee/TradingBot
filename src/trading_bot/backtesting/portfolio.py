from dataclasses import dataclass
from decimal import Decimal

from trading_bot.backtesting.models import (
    Fill,
    OrderSide,
)


class PortfolioError(RuntimeError):
    """Raised when a fill violates portfolio constraints."""


@dataclass(slots=True)
class Position:
    instrument_id: str
    quantity: Decimal = Decimal("0")
    average_entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    @property
    def cost_basis(self) -> Decimal:
        return self.quantity * self.average_entry_price


class Portfolio:
    def __init__(
        self,
        initial_cash: Decimal,
        quote_currency: str = "USDT",
    ) -> None:
        if not initial_cash.is_finite():
            raise ValueError("initial_cash must be finite")

        if initial_cash < 0:
            raise ValueError("initial_cash must not be negative")

        if not quote_currency:
            raise ValueError("quote_currency must not be empty")

        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._quote_currency = quote_currency
        self._positions: dict[str, Position] = {}
        self._fees_paid = Decimal("0")

    @property
    def initial_cash(self) -> Decimal:
        return self._initial_cash

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def quote_currency(self) -> str:
        return self._quote_currency

    @property
    def fees_paid(self) -> Decimal:
        return self._fees_paid

    @property
    def total_realized_pnl(self) -> Decimal:
        return sum(
            (position.realized_pnl for position in self._positions.values()),
            start=Decimal("0"),
        )

    def get_position(
        self,
        instrument_id: str,
    ) -> Position:
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")

        position = self._positions.get(instrument_id)

        if position is None:
            return Position(instrument_id=instrument_id)

        return Position(
            instrument_id=position.instrument_id,
            quantity=position.quantity,
            average_entry_price=(position.average_entry_price),
            realized_pnl=position.realized_pnl,
        )

    def apply_fill(
        self,
        fill: Fill,
    ) -> None:
        if fill.side == OrderSide.BUY:
            self._apply_buy(fill)
        else:
            self._apply_sell(fill)

        self._fees_paid += fill.fee

    def equity(
        self,
        mark_prices: dict[str, Decimal],
    ) -> Decimal:
        total = self._cash

        for (
            instrument_id,
            position,
        ) in self._positions.items():
            if not position.is_open:
                continue

            mark_price = mark_prices.get(instrument_id)

            if mark_price is None:
                raise PortfolioError(f"Missing mark price for open position: {instrument_id}")

            if not mark_price.is_finite() or mark_price <= 0:
                raise PortfolioError(f"Invalid mark price for {instrument_id}: {mark_price}")

            total += position.quantity * mark_price

        return total

    def _apply_buy(
        self,
        fill: Fill,
    ) -> None:
        total_cost = fill.notional + fill.fee

        if total_cost > self._cash:
            raise PortfolioError(
                f"Insufficient cash for buy fill: required={total_cost}, available={self._cash}"
            )

        position = self._positions.setdefault(
            fill.instrument_id,
            Position(instrument_id=fill.instrument_id),
        )

        existing_cost_basis = position.quantity * position.average_entry_price

        new_quantity = position.quantity + fill.quantity

        new_cost_basis = existing_cost_basis + fill.notional + fill.fee

        position.quantity = new_quantity
        position.average_entry_price = new_cost_basis / new_quantity

        self._cash -= total_cost

    def _apply_sell(
        self,
        fill: Fill,
    ) -> None:
        position = self._positions.get(fill.instrument_id)

        if position is None or fill.quantity > position.quantity:
            available_quantity = position.quantity if position is not None else Decimal("0")

            raise PortfolioError(
                "Insufficient position for sell fill: "
                f"required={fill.quantity}, "
                f"available={available_quantity}"
            )

        sale_proceeds = fill.notional - fill.fee

        removed_cost_basis = position.average_entry_price * fill.quantity

        position.realized_pnl += sale_proceeds - removed_cost_basis

        position.quantity -= fill.quantity
        self._cash += sale_proceeds

        if position.quantity == 0:
            position.average_entry_price = Decimal("0")

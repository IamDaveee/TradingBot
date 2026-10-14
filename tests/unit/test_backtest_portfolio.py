from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_bot.backtesting.fees import (
    PercentageFeeModel,
)
from trading_bot.backtesting.models import (
    Fill,
    OrderRequest,
    OrderSide,
)
from trading_bot.backtesting.portfolio import (
    Portfolio,
    PortfolioError,
)

TIMESTAMP = datetime(
    2026,
    6,
    27,
    12,
    0,
    tzinfo=UTC,
)


def create_fill(
    side: OrderSide,
    quantity: str,
    price: str,
    fee: str,
    symbol: str = "BTC-USDT",
) -> Fill:
    return Fill(
        instrument_id=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        timestamp=TIMESTAMP,
    )


def test_fee_model_calculates_percentage() -> None:
    fee_model = PercentageFeeModel(rate=Decimal("0.001"))

    fee = fee_model.calculate(
        quantity=Decimal("2"),
        price=Decimal("100"),
    )

    assert fee == Decimal("0.200")


def test_buy_reduces_cash_and_opens_position() -> None:
    portfolio = Portfolio(initial_cash=Decimal("1000"))

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.BUY,
            quantity="2",
            price="100",
            fee="0.2",
        )
    )

    position = portfolio.get_position("BTC-USDT")

    assert portfolio.cash == Decimal("799.8")
    assert position.quantity == Decimal("2")

    assert position.average_entry_price == Decimal("100.1")

    assert portfolio.fees_paid == Decimal("0.2")


def test_multiple_buys_update_average_cost() -> None:
    portfolio = Portfolio(initial_cash=Decimal("1000"))

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.BUY,
            quantity="1",
            price="100",
            fee="0",
        )
    )

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.BUY,
            quantity="1",
            price="120",
            fee="0",
        )
    )

    position = portfolio.get_position("BTC-USDT")

    assert position.quantity == Decimal("2")

    assert position.average_entry_price == Decimal("110")


def test_sell_realizes_profit() -> None:
    portfolio = Portfolio(initial_cash=Decimal("1000"))

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.BUY,
            quantity="2",
            price="100",
            fee="0",
        )
    )

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.SELL,
            quantity="1",
            price="120",
            fee="0",
        )
    )

    position = portfolio.get_position("BTC-USDT")

    assert portfolio.cash == Decimal("920")
    assert position.quantity == Decimal("1")
    assert position.realized_pnl == Decimal("20")


def test_round_trip_includes_fees() -> None:
    portfolio = Portfolio(initial_cash=Decimal("1000"))

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.BUY,
            quantity="1",
            price="100",
            fee="1",
        )
    )

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.SELL,
            quantity="1",
            price="110",
            fee="1",
        )
    )

    position = portfolio.get_position("BTC-USDT")

    assert portfolio.cash == Decimal("1008")
    assert position.quantity == Decimal("0")
    assert position.realized_pnl == Decimal("8")
    assert portfolio.fees_paid == Decimal("2")


def test_rejects_buy_with_insufficient_cash() -> None:
    portfolio = Portfolio(initial_cash=Decimal("100"))

    with pytest.raises(PortfolioError):
        portfolio.apply_fill(
            create_fill(
                side=OrderSide.BUY,
                quantity="2",
                price="100",
                fee="0",
            )
        )


def test_rejects_sell_above_position() -> None:
    portfolio = Portfolio(initial_cash=Decimal("1000"))

    with pytest.raises(PortfolioError):
        portfolio.apply_fill(
            create_fill(
                side=OrderSide.SELL,
                quantity="1",
                price="100",
                fee="0",
            )
        )


def test_equity_marks_open_positions() -> None:
    portfolio = Portfolio(initial_cash=Decimal("1000"))

    portfolio.apply_fill(
        create_fill(
            side=OrderSide.BUY,
            quantity="2",
            price="100",
            fee="0",
        )
    )

    equity = portfolio.equity(
        {
            "BTC-USDT": Decimal("125"),
        }
    )

    assert equity == Decimal("1050")


def test_order_request_rejects_zero_quantity() -> None:
    with pytest.raises(ValueError):
        OrderRequest(
            instrument_id="BTC-USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0"),
            created_at=TIMESTAMP,
        )

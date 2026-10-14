from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.models import (
    Fill,
    OrderSide,
)
from trading_bot.backtesting.trades import (
    TradeAnalysisError,
    TradeAnalyzer,
    calculate_trade_statistics,
)

BASE_TIME = datetime(
    2026,
    6,
    27,
    12,
    0,
    tzinfo=UTC,
)


def fill(
    side: OrderSide,
    quantity: str,
    price: str,
    fee: str = "0",
    minute: int = 0,
) -> Fill:
    return Fill(
        instrument_id="BTC-USDT",
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        timestamp=(BASE_TIME + timedelta(minutes=minute)),
    )


def test_reconstructs_simple_round_trip() -> None:
    analyzer = TradeAnalyzer()

    trades, open_lots = analyzer.analyze(
        [
            fill(
                OrderSide.BUY,
                quantity="1",
                price="100",
                fee="1",
                minute=0,
            ),
            fill(
                OrderSide.SELL,
                quantity="1",
                price="120",
                fee="1",
                minute=5,
            ),
        ]
    )

    assert len(trades) == 1
    assert open_lots == []

    trade = trades[0]

    assert trade.quantity == Decimal("1")
    assert trade.gross_pnl == Decimal("20")
    assert trade.net_pnl == Decimal("18")
    assert trade.entry_fee == Decimal("1")
    assert trade.exit_fee == Decimal("1")

    assert trade.holding_duration == timedelta(minutes=5)


def test_fifo_matches_multiple_buy_lots() -> None:
    analyzer = TradeAnalyzer()

    trades, open_lots = analyzer.analyze(
        [
            fill(
                OrderSide.BUY,
                quantity="1",
                price="100",
                minute=0,
            ),
            fill(
                OrderSide.BUY,
                quantity="1",
                price="110",
                minute=1,
            ),
            fill(
                OrderSide.SELL,
                quantity="1.5",
                price="120",
                minute=2,
            ),
        ]
    )

    assert len(trades) == 2

    assert trades[0].quantity == Decimal("1")
    assert trades[0].entry_price == Decimal("100")
    assert trades[0].net_pnl == Decimal("20")

    assert trades[1].quantity == Decimal("0.5")
    assert trades[1].entry_price == Decimal("110")
    assert trades[1].net_pnl == Decimal("5.0")

    assert len(open_lots) == 1
    assert open_lots[0].quantity == Decimal("0.5")
    assert open_lots[0].entry_price == Decimal("110")


def test_partially_allocates_fees() -> None:
    analyzer = TradeAnalyzer()

    trades, open_lots = analyzer.analyze(
        [
            fill(
                OrderSide.BUY,
                quantity="2",
                price="100",
                fee="2",
                minute=0,
            ),
            fill(
                OrderSide.SELL,
                quantity="1",
                price="110",
                fee="1",
                minute=1,
            ),
        ]
    )

    assert len(trades) == 1
    assert trades[0].entry_fee == Decimal("1.0")
    assert trades[0].exit_fee == Decimal("1")
    assert trades[0].net_pnl == Decimal("8.0")

    assert len(open_lots) == 1
    assert open_lots[0].quantity == Decimal("1")
    assert open_lots[0].remaining_entry_fee == Decimal("1.0")


def test_rejects_unmatched_sell() -> None:
    analyzer = TradeAnalyzer()

    with pytest.raises(TradeAnalysisError):
        analyzer.analyze(
            [
                fill(
                    OrderSide.SELL,
                    quantity="1",
                    price="100",
                )
            ]
        )


def test_calculates_trade_statistics() -> None:
    analyzer = TradeAnalyzer()

    trades, _ = analyzer.analyze(
        [
            fill(
                OrderSide.BUY,
                quantity="1",
                price="100",
                minute=0,
            ),
            fill(
                OrderSide.SELL,
                quantity="1",
                price="120",
                minute=1,
            ),
            fill(
                OrderSide.BUY,
                quantity="1",
                price="100",
                minute=2,
            ),
            fill(
                OrderSide.SELL,
                quantity="1",
                price="90",
                minute=3,
            ),
        ]
    )

    statistics = calculate_trade_statistics(trades)

    assert statistics.completed_trade_count == 2

    assert statistics.winning_trade_count == 1
    assert statistics.losing_trade_count == 1

    assert statistics.win_rate_percentage == Decimal("50.0")

    assert statistics.total_net_pnl == Decimal("10")

    assert statistics.average_win == Decimal("20")

    assert statistics.average_loss == Decimal("-10")

    assert statistics.profit_factor == Decimal("2")

    assert statistics.best_trade_net_pnl == Decimal("20")

    assert statistics.worst_trade_net_pnl == Decimal("-10")

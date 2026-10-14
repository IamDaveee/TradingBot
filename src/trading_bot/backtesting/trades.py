from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtesting.models import (
    Fill,
    OrderSide,
)


class TradeAnalysisError(RuntimeError):
    """Raised when fills cannot be reconstructed consistently."""


@dataclass(frozen=True, slots=True)
class CompletedTrade:
    instrument_id: str
    quantity: Decimal
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: Decimal
    exit_price: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    return_percentage: Decimal
    holding_duration: timedelta

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def is_loss(self) -> bool:
        return self.net_pnl < 0

    @property
    def is_break_even(self) -> bool:
        return self.net_pnl == 0


@dataclass(frozen=True, slots=True)
class OpenTradeLot:
    instrument_id: str
    quantity: Decimal
    entry_timestamp: datetime
    entry_price: Decimal
    remaining_entry_fee: Decimal


@dataclass(frozen=True, slots=True)
class TradeStatistics:
    completed_trade_count: int
    winning_trade_count: int
    losing_trade_count: int
    break_even_trade_count: int
    win_rate_percentage: Decimal
    total_net_pnl: Decimal
    average_net_pnl: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal | None
    best_trade_net_pnl: Decimal | None
    worst_trade_net_pnl: Decimal | None
    average_holding_duration: timedelta | None


@dataclass(slots=True)
class _MutableOpenLot:
    instrument_id: str
    remaining_quantity: Decimal
    entry_timestamp: datetime
    entry_price: Decimal
    remaining_entry_fee: Decimal


class TradeAnalyzer:
    def analyze(
        self,
        fills: list[Fill],
    ) -> tuple[
        list[CompletedTrade],
        list[OpenTradeLot],
    ]:
        sorted_fills = sorted(
            fills,
            key=lambda fill: (
                fill.timestamp,
                fill.instrument_id,
                fill.side.value,
            ),
        )

        open_lots: dict[
            str,
            deque[_MutableOpenLot],
        ] = {}

        completed_trades: list[CompletedTrade] = []

        for fill in sorted_fills:
            symbol_lots = open_lots.setdefault(
                fill.instrument_id,
                deque(),
            )

            if fill.side == OrderSide.BUY:
                symbol_lots.append(
                    _MutableOpenLot(
                        instrument_id=fill.instrument_id,
                        remaining_quantity=fill.quantity,
                        entry_timestamp=fill.timestamp,
                        entry_price=fill.price,
                        remaining_entry_fee=fill.fee,
                    )
                )
                continue

            self._consume_sell_fill(
                fill=fill,
                lots=symbol_lots,
                completed_trades=completed_trades,
            )

        remaining_lots: list[OpenTradeLot] = []

        for symbol in sorted(open_lots):
            for lot in open_lots[symbol]:
                remaining_lots.append(
                    OpenTradeLot(
                        instrument_id=lot.instrument_id,
                        quantity=lot.remaining_quantity,
                        entry_timestamp=lot.entry_timestamp,
                        entry_price=lot.entry_price,
                        remaining_entry_fee=(lot.remaining_entry_fee),
                    )
                )

        return completed_trades, remaining_lots

    def _consume_sell_fill(
        self,
        fill: Fill,
        lots: deque[_MutableOpenLot],
        completed_trades: list[CompletedTrade],
    ) -> None:
        quantity_remaining = fill.quantity
        exit_fee_remaining = fill.fee

        while quantity_remaining > 0:
            if not lots:
                raise TradeAnalysisError(
                    "Sell fill exceeds available buy quantity: "
                    f"symbol={fill.instrument_id}, "
                    f"remaining={quantity_remaining}"
                )

            lot = lots[0]

            matched_quantity = min(
                quantity_remaining,
                lot.remaining_quantity,
            )

            entry_fee_share = self._proportional_share(
                total_value=lot.remaining_entry_fee,
                part_quantity=matched_quantity,
                total_quantity=lot.remaining_quantity,
            )

            exit_fee_share = self._proportional_share(
                total_value=exit_fee_remaining,
                part_quantity=matched_quantity,
                total_quantity=quantity_remaining,
            )

            entry_notional = matched_quantity * lot.entry_price

            exit_notional = matched_quantity * fill.price

            gross_pnl = exit_notional - entry_notional

            net_pnl = gross_pnl - entry_fee_share - exit_fee_share

            entry_cost = entry_notional + entry_fee_share

            if entry_cost > 0:
                return_percentage = net_pnl / entry_cost * Decimal("100")
            else:
                return_percentage = Decimal("0")

            completed_trades.append(
                CompletedTrade(
                    instrument_id=fill.instrument_id,
                    quantity=matched_quantity,
                    entry_timestamp=lot.entry_timestamp,
                    exit_timestamp=fill.timestamp,
                    entry_price=lot.entry_price,
                    exit_price=fill.price,
                    entry_fee=entry_fee_share,
                    exit_fee=exit_fee_share,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    return_percentage=return_percentage,
                    holding_duration=(fill.timestamp - lot.entry_timestamp),
                )
            )

            lot.remaining_quantity -= matched_quantity

            lot.remaining_entry_fee -= entry_fee_share

            quantity_remaining -= matched_quantity

            exit_fee_remaining -= exit_fee_share

            if lot.remaining_quantity == 0:
                lots.popleft()

    @staticmethod
    def _proportional_share(
        total_value: Decimal,
        part_quantity: Decimal,
        total_quantity: Decimal,
    ) -> Decimal:
        if total_quantity <= 0:
            raise TradeAnalysisError("Cannot allocate value across a non-positive quantity")

        if part_quantity == total_quantity:
            return total_value

        return total_value * part_quantity / total_quantity


def calculate_trade_statistics(
    trades: list[CompletedTrade],
) -> TradeStatistics:
    completed_count = len(trades)

    winning_trades = [trade for trade in trades if trade.is_win]

    losing_trades = [trade for trade in trades if trade.is_loss]

    break_even_trades = [trade for trade in trades if trade.is_break_even]

    total_net_pnl = sum(
        (trade.net_pnl for trade in trades),
        start=Decimal("0"),
    )

    if completed_count > 0:
        average_net_pnl = total_net_pnl / Decimal(completed_count)

        win_rate = Decimal(len(winning_trades)) / Decimal(completed_count) * Decimal("100")

        total_holding_seconds = sum(
            (trade.holding_duration.total_seconds() for trade in trades),
            start=0.0,
        )

        average_holding_duration = timedelta(seconds=(total_holding_seconds / completed_count))
    else:
        average_net_pnl = Decimal("0")
        win_rate = Decimal("0")
        average_holding_duration = None

    total_wins = sum(
        (trade.net_pnl for trade in winning_trades),
        start=Decimal("0"),
    )

    total_losses_absolute = sum(
        (abs(trade.net_pnl) for trade in losing_trades),
        start=Decimal("0"),
    )

    average_win = total_wins / Decimal(len(winning_trades)) if winning_trades else Decimal("0")

    average_loss = (
        -(total_losses_absolute / Decimal(len(losing_trades))) if losing_trades else Decimal("0")
    )

    if total_losses_absolute > 0:
        profit_factor: Decimal | None = total_wins / total_losses_absolute
    elif total_wins > 0:
        profit_factor = None
    else:
        profit_factor = Decimal("0")

    best_trade = (
        max(
            (trade.net_pnl for trade in trades),
        )
        if trades
        else None
    )

    worst_trade = (
        min(
            (trade.net_pnl for trade in trades),
        )
        if trades
        else None
    )

    return TradeStatistics(
        completed_trade_count=completed_count,
        winning_trade_count=len(winning_trades),
        losing_trade_count=len(losing_trades),
        break_even_trade_count=len(break_even_trades),
        win_rate_percentage=win_rate,
        total_net_pnl=total_net_pnl,
        average_net_pnl=average_net_pnl,
        average_win=average_win,
        average_loss=average_loss,
        profit_factor=profit_factor,
        best_trade_net_pnl=best_trade,
        worst_trade_net_pnl=worst_trade,
        average_holding_duration=(average_holding_duration),
    )

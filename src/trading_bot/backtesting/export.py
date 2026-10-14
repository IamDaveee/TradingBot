import csv
from collections.abc import Iterable
from pathlib import Path

from trading_bot.backtesting.execution import (
    EquitySnapshot,
)
from trading_bot.backtesting.models import (
    Fill,
)
from trading_bot.backtesting.trades import (
    CompletedTrade,
)


class BacktestCsvExporter:
    def export_fills(
        self,
        fills: Iterable[Fill],
        path: Path,
    ) -> None:
        self._prepare_parent(path)

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "timestamp",
                    "instrument_id",
                    "side",
                    "quantity",
                    "price",
                    "notional",
                    "fee",
                    "cash_change",
                ]
            )

            for fill in fills:
                writer.writerow(
                    [
                        fill.timestamp.isoformat(),
                        fill.instrument_id,
                        fill.side.value,
                        str(fill.quantity),
                        str(fill.price),
                        str(fill.notional),
                        str(fill.fee),
                        str(fill.cash_change),
                    ]
                )

    def export_trades(
        self,
        trades: Iterable[CompletedTrade],
        path: Path,
    ) -> None:
        self._prepare_parent(path)

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "instrument_id",
                    "quantity",
                    "entry_timestamp",
                    "exit_timestamp",
                    "entry_price",
                    "exit_price",
                    "entry_fee",
                    "exit_fee",
                    "gross_pnl",
                    "net_pnl",
                    "return_percentage",
                    "holding_seconds",
                ]
            )

            for trade in trades:
                writer.writerow(
                    [
                        trade.instrument_id,
                        str(trade.quantity),
                        trade.entry_timestamp.isoformat(),
                        trade.exit_timestamp.isoformat(),
                        str(trade.entry_price),
                        str(trade.exit_price),
                        str(trade.entry_fee),
                        str(trade.exit_fee),
                        str(trade.gross_pnl),
                        str(trade.net_pnl),
                        str(trade.return_percentage),
                        str(trade.holding_duration.total_seconds()),
                    ]
                )

    def export_equity_curve(
        self,
        snapshots: Iterable[EquitySnapshot],
        path: Path,
    ) -> None:
        self._prepare_parent(path)

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "timestamp",
                    "cash",
                    "equity",
                    "realized_pnl",
                    "fees_paid",
                ]
            )

            for snapshot in snapshots:
                writer.writerow(
                    [
                        snapshot.timestamp.isoformat(),
                        str(snapshot.cash),
                        str(snapshot.equity),
                        str(snapshot.realized_pnl),
                        str(snapshot.fees_paid),
                    ]
                )

    @staticmethod
    def _prepare_parent(
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

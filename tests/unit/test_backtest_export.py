import csv
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from trading_bot.backtesting.execution import (
    EquitySnapshot,
)
from trading_bot.backtesting.export import (
    BacktestCsvExporter,
)
from trading_bot.backtesting.models import (
    Fill,
    OrderSide,
)
from trading_bot.backtesting.trades import (
    CompletedTrade,
)

TIMESTAMP = datetime(
    2026,
    6,
    27,
    12,
    0,
    tzinfo=UTC,
)


def read_csv(
    path: Path,
) -> list[list[str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.reader(file))


def test_exports_fills(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fills.csv"

    fill = Fill(
        instrument_id="BTC-USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0.1"),
        timestamp=TIMESTAMP,
    )

    BacktestCsvExporter().export_fills(
        [fill],
        path,
    )

    rows = read_csv(path)

    assert rows[0][0] == "timestamp"
    assert rows[1][1] == "BTC-USDT"
    assert rows[1][2] == "buy"
    assert rows[1][6] == "0.1"


def test_exports_trades(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trades.csv"

    trade = CompletedTrade(
        instrument_id="BTC-USDT",
        quantity=Decimal("1"),
        entry_timestamp=TIMESTAMP,
        exit_timestamp=(TIMESTAMP + timedelta(minutes=5)),
        entry_price=Decimal("100"),
        exit_price=Decimal("110"),
        entry_fee=Decimal("0.1"),
        exit_fee=Decimal("0.1"),
        gross_pnl=Decimal("10"),
        net_pnl=Decimal("9.8"),
        return_percentage=Decimal("9.790209790209790209790209790"),
        holding_duration=timedelta(minutes=5),
    )

    BacktestCsvExporter().export_trades(
        [trade],
        path,
    )

    rows = read_csv(path)

    assert rows[0][0] == "instrument_id"
    assert rows[1][0] == "BTC-USDT"
    assert rows[1][8] == "10"
    assert rows[1][9] == "9.8"
    assert rows[1][11] == "300.0"


def test_exports_equity_curve(
    tmp_path: Path,
) -> None:
    path = tmp_path / "equity.csv"

    snapshot = EquitySnapshot(
        timestamp=TIMESTAMP,
        cash=Decimal("900"),
        equity=Decimal("1000"),
        realized_pnl=Decimal("0"),
        fees_paid=Decimal("0.1"),
    )

    BacktestCsvExporter().export_equity_curve(
        [snapshot],
        path,
    )

    rows = read_csv(path)

    assert rows[0] == [
        "timestamp",
        "cash",
        "equity",
        "realized_pnl",
        "fees_paid",
    ]

    assert rows[1][1] == "900"
    assert rows[1][2] == "1000"

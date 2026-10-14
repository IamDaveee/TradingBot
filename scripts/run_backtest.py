import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

from trading_bot.backtesting.execution import (
    BacktestExecutionEngine,
)
from trading_bot.backtesting.export import (
    BacktestCsvExporter,
)
from trading_bot.backtesting.fees import (
    PercentageFeeModel,
)
from trading_bot.backtesting.metrics import (
    calculate_backtest_metrics,
)
from trading_bot.backtesting.portfolio import (
    Portfolio,
)
from trading_bot.backtesting.runner import (
    StrategyBacktestRunner,
)
from trading_bot.backtesting.trades import (
    TradeAnalyzer,
    calculate_trade_statistics,
)
from trading_bot.backtesting.unified_candle_reader import (
    UnifiedCandleDatasetReader,
)
from trading_bot.market_data.intervals import (
    SUPPORTED_CANDLE_INTERVALS,
    parse_candle_interval,
)
from trading_bot.settings import load_settings
from trading_bot.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)


def parse_date(
    value: str,
) -> date:
    return date.fromisoformat(value)


def positive_decimal(
    value: str,
) -> Decimal:
    parsed = Decimal(value)

    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive decimal")

    return parsed


def non_negative_decimal(
    value: str,
) -> Decimal:
    parsed = Decimal(value)

    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative decimal")

    return parsed


def candle_interval(
    value: str,
) -> str:
    try:
        return parse_candle_interval(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run a moving-average crossover backtest"))

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--start-date",
        type=parse_date,
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
    )

    parser.add_argument(
        "--initial-cash",
        type=positive_decimal,
        default=Decimal("10000"),
    )

    parser.add_argument(
        "--fast-window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--slow-window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--allocation",
        type=positive_decimal,
        default=Decimal("0.95"),
    )

    parser.add_argument(
        "--fee-rate",
        type=non_negative_decimal,
        default=Decimal("0.001"),
    )

    parser.add_argument(
        "--slippage-bps",
        type=non_negative_decimal,
        default=Decimal("0"),
    )

    parser.add_argument(
        "--liquidate-at-end",
        action="store_true",
    )

    parser.add_argument(
        "--export-directory",
        type=Path,
    )
    parser.add_argument(
        "--interval",
        type=candle_interval,
        choices=list(SUPPORTED_CANDLE_INTERVALS),
        default="1m",
        help=("Candle interval used by the strategy. Supported values: 1m, 5m, 15m, 1h"),
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    if arguments.fast_window <= 0:
        raise ValueError("fast-window must be greater than zero")

    if arguments.slow_window <= 0:
        raise ValueError("slow-window must be greater than zero")

    if arguments.fast_window >= (arguments.slow_window):
        raise ValueError("fast-window must be below slow-window")

    if arguments.allocation > Decimal("1"):
        raise ValueError("allocation must not exceed 1")

    _, config = load_settings()

    unified_result = UnifiedCandleDatasetReader(
        ticker_dataset_directory=(config.storage.candle_directory),
        historical_dataset_directory=Path("data/processed/historical_candles"),
        allowed_symbols=set(config.market_data.symbols),
        allowed_intervals=set(SUPPORTED_CANDLE_INTERVALS),
    ).read(
        interval=arguments.interval,
        symbols=[arguments.symbol],
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    candles = unified_result.candles

    print(f"Historical candles loaded: {unified_result.historical_candle_count}")
    print(f"Ticker-derived candles loaded: {unified_result.ticker_candle_count}")
    print(f"Overlapping candles replaced: {unified_result.overlap_count}")

    portfolio = Portfolio(initial_cash=arguments.initial_cash)

    engine = BacktestExecutionEngine(
        portfolio=portfolio,
        fee_model=PercentageFeeModel(rate=arguments.fee_rate),
        slippage_basis_points=(arguments.slippage_bps),
        liquidate_at_end=(arguments.liquidate_at_end),
    )

    strategy = MovingAverageCrossoverStrategy(
        fast_window=(arguments.fast_window),
        slow_window=(arguments.slow_window),
        allocation_fraction=(arguments.allocation),
    )

    runner = StrategyBacktestRunner(
        execution_engine=engine,
        strategy=strategy,
    )
    result = runner.run(candles)

    metrics = calculate_backtest_metrics(
        result=result,
        initial_equity=(arguments.initial_cash),
    )

    completed_trades, open_trade_lots = TradeAnalyzer().analyze(result.fills)

    trade_statistics = calculate_trade_statistics(completed_trades)

    print(f"Candles: {len(candles)}")
    print(f"Fills: {metrics.fill_count}")
    print(f"Rejected orders: {metrics.rejected_order_count}")
    print(f"Unexecuted orders: {metrics.unexecuted_order_count}")
    print(f"Initial equity: {metrics.initial_equity}")
    print(f"Final equity: {metrics.final_equity}")
    print(f"Absolute return: {metrics.absolute_return}")
    print(f"Return percentage: {metrics.return_percentage}%")
    print(f"Maximum drawdown: {metrics.maximum_drawdown}")
    print(f"Maximum drawdown percentage: {metrics.maximum_drawdown_percentage}%")
    print(f"Fees paid: {metrics.total_fees}")

    print(f"Completed trades: {trade_statistics.completed_trade_count}")
    print(f"Open trade lots: {len(open_trade_lots)}")
    print(f"Winning trades: {trade_statistics.winning_trade_count}")
    print(f"Losing trades: {trade_statistics.losing_trade_count}")
    print(f"Break-even trades: {trade_statistics.break_even_trade_count}")
    print(f"Win rate: {trade_statistics.win_rate_percentage}%")
    print(f"Trade net PnL: {trade_statistics.total_net_pnl}")
    print(f"Average trade PnL: {trade_statistics.average_net_pnl}")
    print(f"Average win: {trade_statistics.average_win}")
    print(f"Average loss: {trade_statistics.average_loss}")

    profit_factor_text = (
        str(trade_statistics.profit_factor)
        if trade_statistics.profit_factor is not None
        else "infinite"
    )

    print(f"Profit factor: {profit_factor_text}")

    print(f"Best trade: {trade_statistics.best_trade_net_pnl}")
    print(f"Worst trade: {trade_statistics.worst_trade_net_pnl}")
    print(f"Average holding duration: {trade_statistics.average_holding_duration}")

    if result.fills:
        print("\nFills:")

        for fill in result.fills:
            print(
                f"  {fill.timestamp} "
                f"{fill.side.value.upper()} "
                f"{fill.quantity} "
                f"{fill.instrument_id} "
                f"@ {fill.price} "
                f"fee={fill.fee}"
            )

    if result.rejected_orders:
        print("\nRejected orders:")

        for rejection in result.rejected_orders:
            print(f"  {rejection.rejected_at} {rejection.order.side.value} {rejection.reason}")

    if arguments.export_directory is not None:
        export_directory = arguments.export_directory

        exporter = BacktestCsvExporter()

        exporter.export_fills(
            fills=result.fills,
            path=export_directory / "fills.csv",
        )

        exporter.export_trades(
            trades=completed_trades,
            path=export_directory / "trades.csv",
        )

        exporter.export_equity_curve(
            snapshots=result.equity_curve,
            path=(export_directory / "equity_curve.csv"),
        )

        print(f"\nExported backtest CSV files to: {export_directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

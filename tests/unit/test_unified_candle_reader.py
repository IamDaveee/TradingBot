from datetime import (
    UTC,
    date,
    datetime,
    timedelta,
)
from decimal import Decimal
from pathlib import Path

from trading_bot.backtesting.unified_candle_reader import (
    UnifiedCandleDatasetReader,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)
from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)
from trading_bot.storage.candle_parquet_writer import (
    CandleParquetWriter,
)
from trading_bot.storage.historical_candle_parquet import (
    HistoricalCandleParquetWriter,
)


def historical_candle(
    minute: int,
    close_price: str = "101",
) -> HistoricalCandle:
    start = datetime(
        2026,
        6,
        1,
        12,
        minute,
        tzinfo=UTC,
    )

    return HistoricalCandle(
        instrument_id="BTC-USDT",
        interval="1m",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=Decimal("100"),
        high_price=Decimal("102"),
        low_price=Decimal("99"),
        close_price=Decimal(close_price),
        base_volume=Decimal("10"),
        currency_volume=Decimal("10"),
        quote_volume=Decimal("1000"),
        is_complete=True,
    )


def ticker_candle(
    minute: int,
    close_price: str = "105",
) -> TickerCandle:
    start = datetime(
        2026,
        6,
        1,
        12,
        minute,
        tzinfo=UTC,
    )

    close = Decimal(close_price)

    return TickerCandle(
        instrument_id="BTC-USDT",
        interval_start=start,
        interval_end=(start + timedelta(minutes=1)),
        open_price=Decimal("100"),
        high_price=max(
            Decimal("100"),
            close,
        ),
        low_price=min(
            Decimal("100"),
            close,
        ),
        close_price=close,
        first_bid_price=Decimal("99.9"),
        last_bid_price=(close - Decimal("0.1")),
        first_ask_price=Decimal("100.1"),
        last_ask_price=(close + Decimal("0.1")),
        minimum_spread=Decimal("0.2"),
        maximum_spread=Decimal("0.2"),
        average_spread=Decimal("0.2"),
        observed_size=Decimal("5"),
        tick_count=5,
        first_exchange_timestamp=start,
        last_exchange_timestamp=(start + timedelta(seconds=50)),
        first_received_timestamp=start,
        last_received_timestamp=(start + timedelta(seconds=50)),
    )


def create_reader(
    ticker_directory: Path,
    historical_directory: Path,
) -> UnifiedCandleDatasetReader:
    return UnifiedCandleDatasetReader(
        ticker_dataset_directory=(ticker_directory),
        historical_dataset_directory=(historical_directory),
        allowed_symbols={"BTC-USDT"},
        allowed_intervals={"1m"},
    )


def test_reads_historical_candles(
    tmp_path: Path,
) -> None:
    ticker_directory = tmp_path / "ticker"

    historical_directory = tmp_path / "historical"

    HistoricalCandleParquetWriter(historical_directory).write(
        [
            historical_candle(0),
            historical_candle(1),
        ]
    )

    result = create_reader(
        ticker_directory,
        historical_directory,
    ).read(
        symbols=["BTC-USDT"],
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 1),
    )

    assert result.total_candles == 2
    assert result.historical_candle_count == 2
    assert result.ticker_candle_count == 0
    assert result.overlap_count == 0

    assert result.candles[0].minimum_spread == Decimal("0")

    assert result.candles[0].observed_size == Decimal("10.000000000000000000")


def test_ticker_candle_wins_on_overlap(
    tmp_path: Path,
) -> None:
    ticker_directory = tmp_path / "ticker"

    historical_directory = tmp_path / "historical"

    HistoricalCandleParquetWriter(historical_directory).write(
        [
            historical_candle(
                0,
                close_price="101",
            )
        ]
    )

    CandleParquetWriter(ticker_directory).write(
        [
            ticker_candle(
                0,
                close_price="109",
            )
        ],
        interval_minutes=1,
    )

    result = create_reader(
        ticker_directory,
        historical_directory,
    ).read(symbols=["BTC-USDT"])

    assert result.total_candles == 1
    assert result.overlap_count == 1

    assert result.candles[0].close_price == Decimal("109.000000000000000000")

    assert (
        result.source_for(
            "BTC-USDT",
            result.candles[0].interval_start,
        )
        == "ticker_aggregation"
    )


def test_merges_non_overlapping_sources(
    tmp_path: Path,
) -> None:
    ticker_directory = tmp_path / "ticker"

    historical_directory = tmp_path / "historical"

    HistoricalCandleParquetWriter(historical_directory).write(
        [
            historical_candle(0),
            historical_candle(1),
        ]
    )

    CandleParquetWriter(ticker_directory).write(
        [
            ticker_candle(2),
        ],
        interval_minutes=1,
    )

    result = create_reader(
        ticker_directory,
        historical_directory,
    ).read(symbols=["BTC-USDT"])

    assert result.total_candles == 3

    assert [candle.interval_start.minute for candle in result.candles] == [
        0,
        1,
        2,
    ]


def test_empty_directories_return_empty_result(
    tmp_path: Path,
) -> None:
    result = create_reader(
        tmp_path / "ticker",
        tmp_path / "historical",
    ).read(symbols=["BTC-USDT"])

    assert result.candles == []
    assert result.sources == []
    assert result.total_candles == 0


def test_uses_historical_data_when_ticker_query_is_empty(
    tmp_path: Path,
) -> None:
    ticker_directory = tmp_path / "ticker"

    historical_directory = tmp_path / "historical"

    CandleParquetWriter(ticker_directory).write(
        [
            ticker_candle(0),
        ],
        interval_minutes=1,
    )

    HistoricalCandleParquetWriter(historical_directory).write(
        [
            historical_candle(0),
            historical_candle(1),
        ]
    )

    result = create_reader(
        ticker_directory,
        historical_directory,
    ).read(
        symbols=["BTC-USDT"],
        start_date=date(2026, 6, 2),
        end_date=date(2026, 6, 2),
    )

    assert result.candles == []
    assert result.ticker_candle_count == 0
    assert result.historical_candle_count == 0

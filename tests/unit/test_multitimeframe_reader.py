from datetime import date
from pathlib import Path

from trading_bot.backtesting.unified_candle_reader import (
    UnifiedCandleDatasetReader,
)


def test_reader_accepts_higher_intervals(
    tmp_path: Path,
) -> None:
    reader = UnifiedCandleDatasetReader(
        ticker_dataset_directory=(tmp_path / "ticker"),
        historical_dataset_directory=(tmp_path / "historical"),
        allowed_symbols={"BTC-USDT"},
        allowed_intervals={
            "1m",
            "5m",
            "15m",
            "1h",
        },
    )

    result = reader.read(
        interval="15m",
        symbols=["BTC-USDT"],
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 31),
    )

    assert result.candles == []
    assert result.historical_candle_count == 0
    assert result.ticker_candle_count == 0

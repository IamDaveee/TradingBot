from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow.dataset as ds

from trading_bot.backtesting.candle_loader import (
    CandleTableLoader,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)
from trading_bot.storage.candle_parquet_reader import (
    CandleDatasetError,
    CandleDatasetReader,
)


@dataclass(frozen=True, slots=True)
class UnifiedCandleSource:
    instrument_id: str
    interval_start: datetime
    source: str


@dataclass(frozen=True, slots=True)
class UnifiedCandleReadResult:
    candles: list[TickerCandle]
    sources: list[UnifiedCandleSource]
    historical_candle_count: int
    ticker_candle_count: int
    overlap_count: int

    @property
    def total_candles(self) -> int:
        return len(self.candles)

    def source_for(
        self,
        instrument_id: str,
        interval_start: datetime,
    ) -> str | None:
        for source in self.sources:
            if source.instrument_id == instrument_id and source.interval_start == interval_start:
                return source.source

        return None


class UnifiedCandleDatasetReader:
    def __init__(
        self,
        ticker_dataset_directory: Path,
        historical_dataset_directory: Path,
        allowed_symbols: set[str],
        allowed_intervals: set[str] | None = None,
    ) -> None:
        if not allowed_symbols:
            raise ValueError("allowed_symbols must not be empty")

        self._ticker_dataset_directory = ticker_dataset_directory

        self._historical_dataset_directory = historical_dataset_directory

        self._allowed_symbols = allowed_symbols

        self._allowed_intervals = allowed_intervals or {"1m"}

    def read(
        self,
        interval: str = "1m",
        symbols: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> UnifiedCandleReadResult:
        selected_symbols = sorted(self._allowed_symbols) if symbols is None else symbols

        self._validate_filters(
            interval=interval,
            symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        historical_candles = self._read_historical_candles(
            interval=interval,
            symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        ticker_candles = self._read_ticker_candles(
            interval=interval,
            symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        merged: dict[
            tuple[str, datetime],
            TickerCandle,
        ] = {}

        source_by_key: dict[
            tuple[str, datetime],
            str,
        ] = {}

        for candle in historical_candles:
            key = (
                candle.instrument_id,
                candle.interval_start,
            )

            merged[key] = candle
            source_by_key[key] = "okx_rest"

        overlap_count = 0

        for candle in ticker_candles:
            key = (
                candle.instrument_id,
                candle.interval_start,
            )

            if key in merged:
                overlap_count += 1

            merged[key] = candle
            source_by_key[key] = "ticker_aggregation"

        candles = sorted(
            merged.values(),
            key=lambda candle: (
                candle.interval_start,
                candle.instrument_id,
            ),
        )

        sources = [
            UnifiedCandleSource(
                instrument_id=candle.instrument_id,
                interval_start=(candle.interval_start),
                source=source_by_key[
                    (
                        candle.instrument_id,
                        candle.interval_start,
                    )
                ],
            )
            for candle in candles
        ]

        return UnifiedCandleReadResult(
            candles=candles,
            sources=sources,
            historical_candle_count=len(historical_candles),
            ticker_candle_count=len(ticker_candles),
            overlap_count=overlap_count,
        )

    def _read_ticker_candles(
        self,
        interval: str,
        symbols: list[str],
        start_date: date | None,
        end_date: date | None,
    ) -> list[TickerCandle]:
        if not self._ticker_dataset_directory.exists():
            return []

        reader = CandleDatasetReader(
            dataset_directory=(self._ticker_dataset_directory),
            allowed_symbols=(self._allowed_symbols),
            allowed_intervals=(self._allowed_intervals),
        )

        try:
            table = reader.read(
                interval=interval,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
        except CandleDatasetError as error:
            if str(error) == "The candle dataset query returned no records":
                return []

            raise

        return CandleTableLoader().load(table)

    def _read_historical_candles(
        self,
        interval: str,
        symbols: list[str],
        start_date: date | None,
        end_date: date | None,
    ) -> list[TickerCandle]:
        if not (self._historical_dataset_directory.exists()):
            return []

        dataset = ds.dataset(  # type: ignore[no-untyped-call]
            self._historical_dataset_directory,
            format="parquet",
            partitioning="hive",
        )

        required_columns = {
            "instrument_id",
            "interval_start",
            "interval_end",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "base_volume",
            "is_complete",
            "interval",
            "symbol",
            "date",
        }

        missing_columns = required_columns - set(dataset.schema.names)

        if missing_columns:
            raise ValueError(
                "Historical candle dataset is "
                "missing columns: " + ", ".join(sorted(missing_columns))
            )

        expression = (
            ds.field("interval")  # type: ignore[attr-defined,no-untyped-call]
            == interval
        )

        symbol_expression = None

        for symbol in symbols:
            current = (
                ds.field("symbol")  # type: ignore[attr-defined,no-untyped-call]
                == symbol
            )

            symbol_expression = (
                current if symbol_expression is None else symbol_expression | current
            )

        if symbol_expression is not None:
            expression = expression & symbol_expression

        if start_date is not None:
            expression = expression & (
                ds.field("date")  # type: ignore[attr-defined,no-untyped-call]
                >= start_date.isoformat()
            )

        if end_date is not None:
            expression = expression & (
                ds.field("date")  # type: ignore[attr-defined,no-untyped-call]
                <= end_date.isoformat()
            )

        table = dataset.to_table(
            columns=[
                "instrument_id",
                "interval_start",
                "interval_end",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "base_volume",
                "is_complete",
                "interval",
                "symbol",
                "date",
            ],
            filter=expression,
        )

        candles: list[TickerCandle] = []

        for row_number, record in enumerate(
            table.to_pylist(),
            start=1,
        ):
            try:
                candle = self._historical_record_to_candle(record)
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(f"Invalid historical candle row {row_number}: {error}") from error

            candles.append(candle)

        candles.sort(
            key=lambda candle: (
                candle.interval_start,
                candle.instrument_id,
            )
        )

        return candles

    @staticmethod
    def _historical_record_to_candle(
        record: dict[str, object],
    ) -> TickerCandle:
        instrument_id = UnifiedCandleDatasetReader._require_string(
            record,
            "instrument_id",
        )

        partition_symbol = UnifiedCandleDatasetReader._require_string(
            record,
            "symbol",
        )

        if instrument_id != partition_symbol:
            raise ValueError("instrument_id does not match the symbol partition")

        interval_start = UnifiedCandleDatasetReader._require_datetime(
            record,
            "interval_start",
        )

        interval_end = UnifiedCandleDatasetReader._require_datetime(
            record,
            "interval_end",
        )

        open_price = UnifiedCandleDatasetReader._require_decimal(
            record,
            "open_price",
        )

        high_price = UnifiedCandleDatasetReader._require_decimal(
            record,
            "high_price",
        )

        low_price = UnifiedCandleDatasetReader._require_decimal(
            record,
            "low_price",
        )

        close_price = UnifiedCandleDatasetReader._require_decimal(
            record,
            "close_price",
        )

        base_volume = UnifiedCandleDatasetReader._require_decimal(
            record,
            "base_volume",
        )

        is_complete = record.get("is_complete")

        if not isinstance(is_complete, bool):
            raise TypeError("is_complete must be a bool")

        if not is_complete:
            raise ValueError("Incomplete historical candle cannot be used for backtesting")

        return TickerCandle(
            instrument_id=instrument_id,
            interval_start=interval_start,
            interval_end=interval_end,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            first_bid_price=open_price,
            last_bid_price=close_price,
            first_ask_price=open_price,
            last_ask_price=close_price,
            minimum_spread=Decimal("0"),
            maximum_spread=Decimal("0"),
            average_spread=Decimal("0"),
            observed_size=base_volume,
            tick_count=1,
            first_exchange_timestamp=(interval_start),
            last_exchange_timestamp=(interval_end),
            first_received_timestamp=(interval_start),
            last_received_timestamp=(interval_end),
        )

    def _validate_filters(
        self,
        interval: str,
        symbols: list[str],
        start_date: date | None,
        end_date: date | None,
    ) -> None:
        if interval not in self._allowed_intervals:
            raise ValueError(f"Unsupported interval: {interval}")

        unknown_symbols = set(symbols) - self._allowed_symbols

        if unknown_symbols:
            raise ValueError("Unknown symbols: " + ", ".join(sorted(unknown_symbols)))

        if not symbols:
            raise ValueError("At least one symbol is required")

        if start_date is not None and end_date is not None and end_date < start_date:
            raise ValueError("end_date must not be before start_date")

    @staticmethod
    def _require_string(
        record: dict[str, object],
        field_name: str,
    ) -> str:
        value = record.get(field_name)

        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        return value

    @staticmethod
    def _require_datetime(
        record: dict[str, object],
        field_name: str,
    ) -> datetime:
        value = record.get(field_name)

        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime")

        if value.tzinfo is None:
            raise ValueError(f"{field_name} must contain a timezone")

        return value.astimezone(UTC)

    @staticmethod
    def _require_decimal(
        record: dict[str, object],
        field_name: str,
    ) -> Decimal:
        value = record.get(field_name)

        if not isinstance(value, Decimal):
            raise TypeError(f"{field_name} must be a Decimal")

        return value

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds


class CandleDatasetError(RuntimeError):
    """Raised when a candle dataset query is invalid."""


class CandleDatasetReader:
    REQUIRED_COLUMNS = {
        "instrument_id",
        "interval_start",
        "interval_end",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "first_bid_price",
        "last_bid_price",
        "first_ask_price",
        "last_ask_price",
        "minimum_spread",
        "maximum_spread",
        "average_spread",
        "observed_size",
        "tick_count",
        "first_exchange_timestamp",
        "last_exchange_timestamp",
        "first_received_timestamp",
        "last_received_timestamp",
        "interval",
        "symbol",
        "date",
    }

    def __init__(
        self,
        dataset_directory: Path,
        allowed_symbols: set[str],
        allowed_intervals: set[str] | None = None,
    ) -> None:
        if not allowed_symbols:
            raise ValueError("At least one allowed symbol is required")

        self._dataset_directory = dataset_directory
        self._allowed_symbols = allowed_symbols
        self._allowed_intervals = allowed_intervals if allowed_intervals is not None else {"1m"}

        if not self._allowed_intervals:
            raise ValueError("At least one allowed interval is required")

    def read(
        self,
        interval: str = "1m",
        symbols: Sequence[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        columns: Sequence[str] | None = None,
        sort: bool = True,
    ) -> pa.Table:
        self._validate_directory()
        self._validate_interval(interval)

        selected_symbols = self._validate_symbols(symbols)

        self._validate_date_range(
            start_date=start_date,
            end_date=end_date,
        )

        dataset = ds.dataset(  # type: ignore[no-untyped-call]
            self._dataset_directory,
            format="parquet",
            partitioning="hive",
        )

        available_columns = set(dataset.schema.names)

        missing_columns = self.REQUIRED_COLUMNS - available_columns

        if missing_columns:
            raise CandleDatasetError(
                "Candle dataset is missing required columns: " + ", ".join(sorted(missing_columns))
            )

        selected_columns = self._validate_columns(
            columns=columns,
            available_columns=available_columns,
        )

        filter_expression = self._build_filter_expression(
            interval=interval,
            symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        table = dataset.to_table(
            columns=selected_columns,
            filter=filter_expression,
        )

        if table.num_rows == 0:
            raise CandleDatasetError("The candle dataset query returned no records")

        self._validate_partition_consistency(table)

        if sort:
            table = table.sort_by(
                [
                    (
                        "interval_start",
                        "ascending",
                    ),
                    (
                        "instrument_id",
                        "ascending",
                    ),
                ]
            )

        return table

    def available_symbols(
        self,
        interval: str = "1m",
    ) -> list[str]:
        self._validate_directory()
        self._validate_interval(interval)

        dataset = ds.dataset(  # type: ignore[no-untyped-call]
            self._dataset_directory,
            format="parquet",
            partitioning="hive",
        )

        interval_field = ds.field(  # type: ignore[attr-defined,no-untyped-call]
            "interval"
        )

        table = dataset.to_table(
            columns=["symbol"],
            filter=interval_field == interval,
        )

        values = {str(value) for value in table.column("symbol").to_pylist() if value is not None}

        return sorted(values)

    def available_dates(
        self,
        interval: str = "1m",
        symbol: str | None = None,
    ) -> list[date]:
        self._validate_directory()
        self._validate_interval(interval)

        dataset = ds.dataset(  # type: ignore[no-untyped-call]
            self._dataset_directory,
            format="parquet",
            partitioning="hive",
        )

        interval_field = ds.field(  # type: ignore[attr-defined,no-untyped-call]
            "interval"
        )

        filter_expression: Any = interval_field == interval

        if symbol is not None:
            self._validate_symbols([symbol])

            symbol_field = ds.field(  # type: ignore[attr-defined,no-untyped-call]
                "symbol"
            )

            filter_expression = filter_expression & (symbol_field == symbol)

        table = dataset.to_table(
            columns=["date"],
            filter=filter_expression,
        )

        values: set[date] = set()

        for value in table.column("date").to_pylist():
            if value is None:
                continue

            if isinstance(value, date):
                values.add(value)
            else:
                values.add(date.fromisoformat(str(value)))

        return sorted(values)

    def _validate_directory(self) -> None:
        if not self._dataset_directory.exists():
            raise FileNotFoundError(
                f"Candle dataset directory does not exist: {self._dataset_directory}"
            )

        if not any(self._dataset_directory.rglob("*.parquet")):
            raise CandleDatasetError("Candle dataset contains no Parquet files")

    def _validate_interval(
        self,
        interval: str,
    ) -> None:
        if interval not in self._allowed_intervals:
            raise ValueError(f"Unsupported candle interval: {interval}")

    def _validate_symbols(
        self,
        symbols: Sequence[str] | None,
    ) -> list[str]:
        if symbols is None:
            return sorted(self._allowed_symbols)

        if not symbols:
            raise ValueError("Symbols must not be empty")

        selected_symbols = list(dict.fromkeys(symbols))

        unknown_symbols = set(selected_symbols) - self._allowed_symbols

        if unknown_symbols:
            raise ValueError("Unsupported symbols: " + ", ".join(sorted(unknown_symbols)))

        return selected_symbols

    @staticmethod
    def _validate_date_range(
        start_date: date | None,
        end_date: date | None,
    ) -> None:
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start_date must not be after end_date")

    @staticmethod
    def _validate_columns(
        columns: Sequence[str] | None,
        available_columns: set[str],
    ) -> list[str] | None:
        if columns is None:
            return None

        if not columns:
            raise ValueError("Columns must not be empty")

        selected_columns = list(dict.fromkeys(columns))

        unknown_columns = set(selected_columns) - available_columns

        if unknown_columns:
            raise ValueError("Unknown columns: " + ", ".join(sorted(unknown_columns)))

        validation_columns = [
            "instrument_id",
            "symbol",
            "interval",
            "date",
            "interval_start",
            "interval_end",
        ]

        for column in validation_columns:
            if column not in selected_columns:
                selected_columns.append(column)

        return selected_columns

    @staticmethod
    def _build_filter_expression(
        interval: str,
        symbols: Sequence[str],
        start_date: date | None,
        end_date: date | None,
    ) -> Any:
        interval_field = ds.field(  # type: ignore[attr-defined,no-untyped-call]
            "interval"
        )

        symbol_field = ds.field(  # type: ignore[attr-defined,no-untyped-call]
            "symbol"
        )

        date_field = ds.field(  # type: ignore[attr-defined,no-untyped-call]
            "date"
        )

        expression = (interval_field == interval) & symbol_field.isin(list(symbols))

        if start_date is not None:
            expression = expression & (date_field >= start_date.isoformat())

        if end_date is not None:
            expression = expression & (date_field <= end_date.isoformat())

        return expression

    @staticmethod
    def _validate_partition_consistency(
        table: pa.Table,
    ) -> None:
        instrument_ids = table.column("instrument_id").to_pylist()

        symbols = table.column("symbol").to_pylist()

        intervals = table.column("interval").to_pylist()

        interval_starts = table.column("interval_start").to_pylist()

        partition_dates = table.column("date").to_pylist()

        for row_number, values in enumerate(
            zip(
                instrument_ids,
                symbols,
                intervals,
                interval_starts,
                partition_dates,
                strict=True,
            ),
            start=1,
        ):
            (
                instrument_id,
                symbol,
                interval,
                interval_start,
                partition_date,
            ) = values

            if instrument_id != symbol:
                raise CandleDatasetError(
                    "Partition symbol mismatch at row "
                    f"{row_number}: "
                    f"instrument_id={instrument_id}, "
                    f"symbol={symbol}"
                )

            if interval != "1m":
                raise CandleDatasetError(
                    f"Unexpected interval partition at row {row_number}: {interval}"
                )

            if interval_start is not None and str(interval_start.date()) != str(partition_date):
                raise CandleDatasetError(
                    "Partition date mismatch at row "
                    f"{row_number}: "
                    f"interval_start={interval_start}, "
                    f"date={partition_date}"
                )

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds


class TickerDatasetError(RuntimeError):
    """Raised when a ticker dataset query is invalid."""


class TickerDatasetReader:
    REQUIRED_COLUMNS = {
        "instrument_id",
        "last_price",
        "last_size",
        "best_ask_price",
        "best_ask_size",
        "best_bid_price",
        "best_bid_size",
        "spread",
        "spread_percentage",
        "exchange_timestamp",
        "received_timestamp",
        "written_timestamp",
        "symbol",
        "date",
    }

    def __init__(
        self,
        dataset_directory: Path,
        allowed_symbols: set[str],
    ) -> None:
        if not allowed_symbols:
            raise ValueError("At least one allowed symbol is required")

        self._dataset_directory = dataset_directory
        self._allowed_symbols = allowed_symbols

    def read(
        self,
        symbols: Sequence[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        columns: Sequence[str] | None = None,
        sort: bool = True,
    ) -> pa.Table:
        self._validate_directory()

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

        missing_required_columns = self.REQUIRED_COLUMNS - available_columns

        if missing_required_columns:
            raise TickerDatasetError(
                "Dataset is missing required columns: "
                + ", ".join(sorted(missing_required_columns))
            )

        selected_columns = self._validate_columns(
            columns=columns,
            available_columns=available_columns,
        )

        filter_expression = self._build_filter_expression(
            symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        table = dataset.to_table(
            columns=selected_columns,
            filter=filter_expression,
        )

        if table.num_rows == 0:
            raise TickerDatasetError("The dataset query returned no records")

        self._validate_partition_consistency(table)

        if sort:
            table = self._sort_table(table)

        return table

    def available_symbols(self) -> list[str]:
        self._validate_directory()

        dataset = ds.dataset(  # type: ignore[no-untyped-call]
            self._dataset_directory,
            format="parquet",
            partitioning="hive",
        )

        table = dataset.to_table(columns=["symbol"])

        values = {str(value) for value in table.column("symbol").to_pylist() if value is not None}

        return sorted(values)

    def available_dates(
        self,
        symbol: str | None = None,
    ) -> list[date]:
        self._validate_directory()

        dataset = ds.dataset(  # type: ignore[no-untyped-call]
            self._dataset_directory,
            format="parquet",
            partitioning="hive",
        )

        filter_expression: Any = None

        if symbol is not None:
            self._validate_symbols([symbol])

            filter_expression = (
                ds.field(  # type: ignore[attr-defined,no-untyped-call]
                    "symbol"
                )
                == symbol
            )

        table = dataset.to_table(
            columns=["date"],
            filter=filter_expression,
        )

        parsed_dates: set[date] = set()

        for value in table.column("date").to_pylist():
            if value is None:
                continue

            if isinstance(value, date):
                parsed_dates.add(value)
            else:
                parsed_dates.add(date.fromisoformat(str(value)))

        return sorted(parsed_dates)

    def _validate_directory(self) -> None:
        if not self._dataset_directory.exists():
            raise FileNotFoundError(
                f"Ticker dataset directory does not exist: {self._dataset_directory}"
            )

        if not any(self._dataset_directory.rglob("*.parquet")):
            raise TickerDatasetError("Ticker dataset contains no Parquet files")

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
            "received_timestamp",
            "exchange_timestamp",
        ]

        for column in validation_columns:
            if column not in selected_columns:
                selected_columns.append(column)

        return selected_columns

    @staticmethod
    def _build_filter_expression(
        symbols: Sequence[str],
        start_date: date | None,
        end_date: date | None,
    ) -> Any:
        expression = ds.field(  # type: ignore[attr-defined,no-untyped-call]
            "symbol"
        ).isin(list(symbols))

        if start_date is not None:
            expression = expression & (
                ds.field(  # type: ignore[attr-defined,no-untyped-call]
                    "date"
                )
                >= start_date.isoformat()
            )

        if end_date is not None:
            expression = expression & (
                ds.field(  # type: ignore[attr-defined,no-untyped-call]
                    "date"
                )
                <= end_date.isoformat()
            )

        return expression

    @staticmethod
    def _validate_partition_consistency(
        table: pa.Table,
    ) -> None:
        instrument_ids = table.column("instrument_id").to_pylist()

        partition_symbols = table.column("symbol").to_pylist()

        for row_number, (
            instrument_id,
            partition_symbol,
        ) in enumerate(
            zip(
                instrument_ids,
                partition_symbols,
                strict=True,
            ),
            start=1,
        ):
            if instrument_id != partition_symbol:
                raise TickerDatasetError(
                    "Partition symbol mismatch at row "
                    f"{row_number}: "
                    f"instrument_id={instrument_id}, "
                    f"symbol={partition_symbol}"
                )

    @staticmethod
    def _sort_table(
        table: pa.Table,
    ) -> pa.Table:
        return table.sort_by(
            [
                (
                    "received_timestamp",
                    "ascending",
                ),
                (
                    "exchange_timestamp",
                    "ascending",
                ),
                (
                    "instrument_id",
                    "ascending",
                ),
            ]
        )

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import (
    ROUND_HALF_EVEN,
    Decimal,
    InvalidOperation,
)
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)

_DECIMAL_QUANTIZER = Decimal("0.000000000000000001")


HISTORICAL_CANDLE_SCHEMA = pa.schema(
    [
        pa.field(
            "instrument_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "interval_start",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "interval_end",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "open_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "high_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "low_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "close_price",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "base_volume",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "currency_volume",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "quote_volume",
            pa.decimal128(38, 18),
            nullable=False,
        ),
        pa.field(
            "is_complete",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "source",
            pa.string(),
            nullable=False,
        ),
    ]
)


@dataclass(frozen=True, slots=True)
class HistoricalWriteResult:
    received_candles: int
    written_candles: int
    existing_candles: int
    duplicate_candles: int
    excluded_candles: int
    partition_count: int
    written_files: list[Path]


class HistoricalCandleParquetWriter:
    def __init__(
        self,
        dataset_directory: Path,
    ) -> None:
        self._dataset_directory = dataset_directory

    def write(
        self,
        candles: list[HistoricalCandle],
        excluded_timestamps: set[tuple[str, datetime]] | None = None,
    ) -> HistoricalWriteResult:
        excluded = excluded_timestamps or set()

        grouped: dict[
            tuple[str, str, date],
            list[HistoricalCandle],
        ] = defaultdict(list)

        duplicate_count = 0
        excluded_count = 0

        incoming_keys: set[tuple[str, datetime]] = set()

        for candle in candles:
            key = (
                candle.instrument_id,
                candle.interval_start,
            )

            if key in excluded:
                excluded_count += 1
                continue

            if key in incoming_keys:
                duplicate_count += 1
                continue

            incoming_keys.add(key)

            partition_key = (
                candle.interval,
                candle.instrument_id,
                candle.interval_start.date(),
            )

            grouped[partition_key].append(candle)

        existing_total = 0
        written_total = 0
        written_files: list[Path] = []

        for (
            interval,
            symbol,
            partition_date,
        ), partition_candles in sorted(grouped.items()):
            partition_directory = (
                self._dataset_directory
                / f"interval={interval}"
                / f"symbol={symbol}"
                / f"date={partition_date.isoformat()}"
            )

            target_path = partition_directory / "part-history.parquet"

            existing_candles = self._read_existing_partition(partition_directory)

            existing_total += len(existing_candles)

            merged: dict[
                datetime,
                HistoricalCandle,
            ] = {candle.interval_start: candle for candle in existing_candles}

            for candle in partition_candles:
                if candle.interval_start in merged:
                    duplicate_count += 1

                # The newly downloaded candle wins.
                merged[candle.interval_start] = candle

            merged_candles = sorted(
                merged.values(),
                key=lambda candle: candle.interval_start,
            )

            self._write_partition_atomically(
                candles=merged_candles,
                partition_directory=(partition_directory),
                target_path=target_path,
            )

            written_total += len(merged_candles)

            written_files.append(target_path)

        return HistoricalWriteResult(
            received_candles=len(candles),
            written_candles=written_total,
            existing_candles=existing_total,
            duplicate_candles=duplicate_count,
            excluded_candles=excluded_count,
            partition_count=len(grouped),
            written_files=written_files,
        )

    def _read_existing_partition(
        self,
        partition_directory: Path,
    ) -> list[HistoricalCandle]:
        if not partition_directory.exists():
            return []

        parquet_files = sorted(partition_directory.glob("*.parquet"))

        if not parquet_files:
            return []

        by_timestamp: dict[
            datetime,
            HistoricalCandle,
        ] = {}

        interval = self._partition_value(
            partition_directory.parent.parent.name,
            "interval",
        )

        symbol = self._partition_value(
            partition_directory.parent.name,
            "symbol",
        )

        for path in parquet_files:
            # ParquetFile avoids Hive partition-column
            # inference when reading one physical file.
            table = pq.ParquetFile(  # type: ignore[no-untyped-call]
                path
            ).read()

            self._validate_schema(table)

            for record in table.to_pylist():
                candle = self._record_to_candle(
                    record=record,
                    interval=interval,
                    symbol=symbol,
                )

                by_timestamp[candle.interval_start] = candle

        return sorted(
            by_timestamp.values(),
            key=lambda candle: candle.interval_start,
        )

    def _write_partition_atomically(
        self,
        candles: list[HistoricalCandle],
        partition_directory: Path,
        target_path: Path,
    ) -> None:
        partition_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = partition_directory / (f".part-history-{uuid4().hex}.tmp.parquet")

        table = pa.Table.from_pylist(
            [self._candle_to_record(candle) for candle in candles],
            schema=HISTORICAL_CANDLE_SCHEMA,
        )

        try:
            pq.write_table(  # type: ignore[no-untyped-call]
                table,
                temporary_path,
                compression="zstd",
            )

            os.replace(
                temporary_path,
                target_path,
            )

            self._remove_obsolete_files(
                partition_directory=(partition_directory),
                target_path=target_path,
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _remove_obsolete_files(
        partition_directory: Path,
        target_path: Path,
    ) -> None:
        for path in partition_directory.glob("*.parquet"):
            if path != target_path:
                path.unlink()

    @staticmethod
    def _candle_to_record(
        candle: HistoricalCandle,
    ) -> dict[str, object]:
        return {
            "instrument_id": (candle.instrument_id),
            "interval_start": (candle.interval_start.astimezone(UTC)),
            "interval_end": (candle.interval_end.astimezone(UTC)),
            "open_price": (HistoricalCandleParquetWriter._normalize_decimal(candle.open_price)),
            "high_price": (HistoricalCandleParquetWriter._normalize_decimal(candle.high_price)),
            "low_price": (HistoricalCandleParquetWriter._normalize_decimal(candle.low_price)),
            "close_price": (HistoricalCandleParquetWriter._normalize_decimal(candle.close_price)),
            "base_volume": (HistoricalCandleParquetWriter._normalize_decimal(candle.base_volume)),
            "currency_volume": (
                HistoricalCandleParquetWriter._normalize_decimal(candle.currency_volume)
            ),
            "quote_volume": (HistoricalCandleParquetWriter._normalize_decimal(candle.quote_volume)),
            "is_complete": candle.is_complete,
            "source": "okx_rest",
        }

    @staticmethod
    def _record_to_candle(
        record: dict[str, object],
        interval: str,
        symbol: str,
    ) -> HistoricalCandle:
        instrument_id = record.get("instrument_id")

        if instrument_id != symbol:
            raise ValueError(
                "Historical candle instrument does "
                "not match its partition: "
                f"record={instrument_id}, "
                f"partition={symbol}"
            )

        interval_start = record.get("interval_start")

        interval_end = record.get("interval_end")

        if not isinstance(
            interval_start,
            datetime,
        ):
            raise TypeError("interval_start must be a datetime")

        if not isinstance(
            interval_end,
            datetime,
        ):
            raise TypeError("interval_end must be a datetime")

        return HistoricalCandle(
            instrument_id=symbol,
            interval=interval,
            interval_start=interval_start,
            interval_end=interval_end,
            open_price=(
                HistoricalCandleParquetWriter._require_decimal(
                    record,
                    "open_price",
                )
            ),
            high_price=(
                HistoricalCandleParquetWriter._require_decimal(
                    record,
                    "high_price",
                )
            ),
            low_price=(
                HistoricalCandleParquetWriter._require_decimal(
                    record,
                    "low_price",
                )
            ),
            close_price=(
                HistoricalCandleParquetWriter._require_decimal(
                    record,
                    "close_price",
                )
            ),
            base_volume=(
                HistoricalCandleParquetWriter._require_decimal(
                    record,
                    "base_volume",
                )
            ),
            currency_volume=(
                HistoricalCandleParquetWriter._require_decimal(
                    record,
                    "currency_volume",
                )
            ),
            quote_volume=(
                HistoricalCandleParquetWriter._require_decimal(
                    record,
                    "quote_volume",
                )
            ),
            is_complete=bool(record.get("is_complete")),
        )

    @staticmethod
    def _require_decimal(
        record: dict[str, object],
        field_name: str,
    ) -> Decimal:
        value = record.get(field_name)

        if not isinstance(value, Decimal):
            raise TypeError(f"{field_name} must be a Decimal")

        return value

    @staticmethod
    def _normalize_decimal(
        value: Decimal,
    ) -> Decimal:
        if not value.is_finite():
            raise ValueError("Decimal value must be finite")

        try:
            return value.quantize(
                _DECIMAL_QUANTIZER,
                rounding=ROUND_HALF_EVEN,
            )
        except InvalidOperation as error:
            raise ValueError("Decimal value cannot fit into decimal128(38, 18)") from error

    @staticmethod
    def _validate_schema(
        table: pa.Table,
    ) -> None:
        required = {field.name for field in (HISTORICAL_CANDLE_SCHEMA)}

        missing = required - set(table.column_names)

        if missing:
            raise ValueError(
                "Historical candle Parquet file is missing columns: " + ", ".join(sorted(missing))
            )

    @staticmethod
    def _partition_value(
        directory_name: str,
        expected_key: str,
    ) -> str:
        prefix = f"{expected_key}="

        if not directory_name.startswith(prefix):
            raise ValueError(f"Invalid historical candle partition: {directory_name}")

        value = directory_name[len(prefix) :]

        if not value:
            raise ValueError(f"Historical candle partition {expected_key} must not be empty")

        return value

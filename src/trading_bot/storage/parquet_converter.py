import json
from collections import defaultdict
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import structlog


class TickerParquetConversionError(RuntimeError):
    """Raised when ticker data cannot be converted safely."""


class TickerParquetConverter:
    _schema = pa.schema(
        [
            pa.field("instrument_id", pa.string(), nullable=False),
            pa.field(
                "last_price",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "last_size",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "best_ask_price",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "best_ask_size",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "best_bid_price",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "best_bid_size",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "spread",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "spread_percentage",
                pa.float64(),
                nullable=False,
            ),
            pa.field(
                "exchange_timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "received_timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "written_timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
        ]
    )

    def __init__(
        self,
        output_directory: Path,
        allowed_symbols: set[str],
    ) -> None:
        if not allowed_symbols:
            raise ValueError("At least one allowed symbol is required")

        self._output_directory = output_directory
        self._allowed_symbols = allowed_symbols
        self._logger = structlog.get_logger("ticker_parquet_converter")

    def convert_file(
        self,
        input_file: Path,
        overwrite: bool = False,
    ) -> list[Path]:
        if not input_file.exists():
            raise FileNotFoundError(f"Ticker JSONL file does not exist: {input_file}")

        grouped_records: dict[
            tuple[str, date],
            list[dict[str, object]],
        ] = defaultdict(list)

        with input_file.open(
            mode="r",
            encoding="utf-8",
        ) as source:
            for line_number, line in enumerate(
                source,
                start=1,
            ):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    raw_record: Any = json.loads(stripped_line)
                except json.JSONDecodeError as error:
                    raise TickerParquetConversionError(
                        f"Invalid JSON on line {line_number}: {error}"
                    ) from error

                if not isinstance(raw_record, dict):
                    raise TickerParquetConversionError(f"Line {line_number} is not a JSON object")

                parsed_record = self._parse_record(
                    raw_record,
                    line_number,
                )

                symbol = str(parsed_record["instrument_id"])
                received_timestamp = parsed_record["received_timestamp"]

                if not isinstance(
                    received_timestamp,
                    datetime,
                ):
                    raise TickerParquetConversionError("Parsed received timestamp is invalid")

                partition_key = (
                    symbol,
                    received_timestamp.date(),
                )

                grouped_records[partition_key].append(parsed_record)

        if not grouped_records:
            raise TickerParquetConversionError(f"No ticker records found in {input_file}")

        written_files: list[Path] = []

        for (
            symbol,
            partition_date,
        ), records in grouped_records.items():
            output_file = self._build_output_path(
                input_file=input_file,
                symbol=symbol,
                partition_date=partition_date,
            )

            if output_file.exists() and not overwrite:
                raise FileExistsError(
                    f"Parquet output already exists: "
                    f"{output_file}. Use overwrite=True "
                    "to replace it."
                )

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            records.sort(
                key=lambda record: (
                    record["received_timestamp"],
                    record["exchange_timestamp"],
                )
            )

            table = pa.Table.from_pylist(
                records,
                schema=self._schema,
            )

            pq.write_table(  # type: ignore[no-untyped-call]
                table,
                output_file,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )

            written_files.append(output_file)

            self._logger.info(
                "ticker_parquet_file_written",
                path=str(output_file),
                symbol=symbol,
                date=partition_date.isoformat(),
                record_count=table.num_rows,
                byte_size=output_file.stat().st_size,
            )

        return written_files

    def _build_output_path(
        self,
        input_file: Path,
        symbol: str,
        partition_date: date,
    ) -> Path:
        return (
            self._output_directory
            / f"symbol={symbol}"
            / f"date={partition_date.isoformat()}"
            / f"part-{input_file.stem}.parquet"
        )

    def _parse_record(
        self,
        record: dict[str, Any],
        line_number: int,
    ) -> dict[str, object]:
        required_fields = {
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
        }

        missing_fields = required_fields - record.keys()

        if missing_fields:
            raise TickerParquetConversionError(
                f"Line {line_number} is missing fields: " + ", ".join(sorted(missing_fields))
            )

        symbol = str(record["instrument_id"])

        if symbol not in self._allowed_symbols:
            raise TickerParquetConversionError(f"Unexpected symbol on line {line_number}: {symbol}")

        try:
            return {
                "instrument_id": symbol,
                "last_price": self._decimal(record["last_price"]),
                "last_size": self._decimal(record["last_size"]),
                "best_ask_price": self._decimal(record["best_ask_price"]),
                "best_ask_size": self._decimal(record["best_ask_size"]),
                "best_bid_price": self._decimal(record["best_bid_price"]),
                "best_bid_size": self._decimal(record["best_bid_size"]),
                "spread": self._decimal(record["spread"]),
                "spread_percentage": float(record["spread_percentage"]),
                "exchange_timestamp": (self._datetime(record["exchange_timestamp"])),
                "received_timestamp": (self._datetime(record["received_timestamp"])),
                "written_timestamp": self._datetime(record["written_timestamp"]),
            }
        except (
            InvalidOperation,
            TypeError,
            ValueError,
        ) as error:
            raise TickerParquetConversionError(
                f"Invalid value on line {line_number}: {error}"
            ) from error

    @staticmethod
    def _decimal(value: object) -> Decimal:
        parsed_value = Decimal(str(value))

        if not parsed_value.is_finite():
            raise InvalidOperation("Decimal value must be finite")

        return parsed_value

    @staticmethod
    def _datetime(value: object) -> datetime:
        parsed_value = datetime.fromisoformat(str(value))

        if parsed_value.tzinfo is None:
            raise ValueError("Timestamp must contain a timezone")

        return parsed_value.astimezone(UTC)

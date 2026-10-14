from collections import defaultdict
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from trading_bot.market_data.candle import TickerCandle


class CandleParquetWriteError(RuntimeError):
    """Raised when candle data cannot be written safely."""


class CandleParquetWriter:
    _DECIMAL_SCALE = 18
    _DECIMAL_QUANTIZER = Decimal("0.000000000000000001")
    _schema = pa.schema(
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
                "first_bid_price",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "last_bid_price",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "first_ask_price",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "last_ask_price",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "minimum_spread",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "maximum_spread",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "average_spread",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "observed_size",
                pa.decimal128(38, 18),
                nullable=False,
            ),
            pa.field(
                "tick_count",
                pa.int64(),
                nullable=False,
            ),
            pa.field(
                "first_exchange_timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "last_exchange_timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "first_received_timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "last_received_timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
        ]
    )

    def __init__(
        self,
        output_directory: Path,
    ) -> None:
        self._output_directory = output_directory
        self._logger = structlog.get_logger("candle_parquet_writer")

    def write(
        self,
        candles: list[TickerCandle],
        interval_minutes: int,
        overwrite: bool = False,
    ) -> list[Path]:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")

        if not candles:
            raise CandleParquetWriteError("No candles were supplied")

        interval_name = f"{interval_minutes}m"

        grouped_candles: dict[
            tuple[str, date],
            list[TickerCandle],
        ] = defaultdict(list)

        for candle in candles:
            grouped_candles[
                (
                    candle.instrument_id,
                    candle.interval_start.date(),
                )
            ].append(candle)

        written_files: list[Path] = []

        for (
            symbol,
            partition_date,
        ), partition_candles in grouped_candles.items():
            output_file = self._build_output_path(
                symbol=symbol,
                partition_date=partition_date,
                interval_name=interval_name,
            )

            if output_file.exists() and not overwrite:
                raise FileExistsError(
                    f"Candle output already exists: "
                    f"{output_file}. Use overwrite=True "
                    "to replace it."
                )

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            partition_candles.sort(
                key=lambda candle: (
                    candle.interval_start,
                    candle.instrument_id,
                )
            )

            records = [
                self._to_record(
                    candle=candle,
                )
                for candle in partition_candles
            ]

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
                "candle_parquet_file_written",
                path=str(output_file),
                symbol=symbol,
                date=partition_date.isoformat(),
                interval=interval_name,
                candle_count=table.num_rows,
                byte_size=output_file.stat().st_size,
            )

        return written_files

    def _build_output_path(
        self,
        symbol: str,
        partition_date: date,
        interval_name: str,
    ) -> Path:
        return (
            self._output_directory
            / f"interval={interval_name}"
            / f"symbol={symbol}"
            / f"date={partition_date.isoformat()}"
            / (f"part-candles-{partition_date.isoformat()}.parquet")
        )

    @classmethod
    def _to_record(
        cls,
        candle: TickerCandle,
    ) -> dict[str, object]:
        return {
            "instrument_id": candle.instrument_id,
            "interval_start": candle.interval_start,
            "interval_end": candle.interval_end,
            "open_price": cls._normalize_decimal(
                candle.open_price,
                "open_price",
            ),
            "high_price": cls._normalize_decimal(
                candle.high_price,
                "high_price",
            ),
            "low_price": cls._normalize_decimal(
                candle.low_price,
                "low_price",
            ),
            "close_price": cls._normalize_decimal(
                candle.close_price,
                "close_price",
            ),
            "first_bid_price": cls._normalize_decimal(
                candle.first_bid_price,
                "first_bid_price",
            ),
            "last_bid_price": cls._normalize_decimal(
                candle.last_bid_price,
                "last_bid_price",
            ),
            "first_ask_price": cls._normalize_decimal(
                candle.first_ask_price,
                "first_ask_price",
            ),
            "last_ask_price": cls._normalize_decimal(
                candle.last_ask_price,
                "last_ask_price",
            ),
            "minimum_spread": cls._normalize_decimal(
                candle.minimum_spread,
                "minimum_spread",
            ),
            "maximum_spread": cls._normalize_decimal(
                candle.maximum_spread,
                "maximum_spread",
            ),
            "average_spread": cls._normalize_decimal(
                candle.average_spread,
                "average_spread",
            ),
            "observed_size": cls._normalize_decimal(
                candle.observed_size,
                "observed_size",
            ),
            "tick_count": candle.tick_count,
            "first_exchange_timestamp": (candle.first_exchange_timestamp),
            "last_exchange_timestamp": (candle.last_exchange_timestamp),
            "first_received_timestamp": (candle.first_received_timestamp),
            "last_received_timestamp": (candle.last_received_timestamp),
        }

    @classmethod
    def _normalize_decimal(
        cls,
        value: Decimal,
        field_name: str,
    ) -> Decimal:
        if not value.is_finite():
            raise CandleParquetWriteError(f"{field_name} must be finite")

        try:
            normalized = value.quantize(
                cls._DECIMAL_QUANTIZER,
                rounding=ROUND_HALF_EVEN,
            )
        except InvalidOperation as error:
            raise CandleParquetWriteError(
                f"{field_name} cannot be represented as decimal128(38, 18)"
            ) from error

        integer_digits = max(
            normalized.adjusted() + 1,
            0,
        )

        if integer_digits > 20:
            raise CandleParquetWriteError(
                f"{field_name} exceeds the supported decimal128(38, 18) range"
            )

        return normalized

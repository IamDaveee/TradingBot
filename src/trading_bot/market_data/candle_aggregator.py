from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pyarrow as pa

from trading_bot.market_data.candle import (
    TickerCandle,
)


class CandleAggregationError(RuntimeError):
    """Raised when ticker rows cannot be aggregated."""


class TickerCandleAggregator:
    REQUIRED_COLUMNS = {
        "instrument_id",
        "last_price",
        "last_size",
        "best_bid_price",
        "best_ask_price",
        "spread",
        "exchange_timestamp",
        "received_timestamp",
    }

    def aggregate(
        self,
        table: pa.Table,
        interval_minutes: int = 1,
    ) -> list[TickerCandle]:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")

        missing_columns = self.REQUIRED_COLUMNS - set(table.column_names)

        if missing_columns:
            raise CandleAggregationError(
                "Ticker table is missing required columns: " + ", ".join(sorted(missing_columns))
            )

        if table.num_rows == 0:
            return []

        records = table.to_pylist()

        grouped_records: dict[
            tuple[str, datetime],
            list[dict[str, object]],
        ] = defaultdict(list)

        for row_number, record in enumerate(
            records,
            start=1,
        ):
            parsed_record = self._validate_record(
                record=record,
                row_number=row_number,
            )

            instrument_id = parsed_record["instrument_id"]
            received_timestamp = parsed_record["received_timestamp"]

            if not isinstance(
                instrument_id,
                str,
            ):
                raise CandleAggregationError("Parsed instrument_id is invalid")

            if not isinstance(
                received_timestamp,
                datetime,
            ):
                raise CandleAggregationError("Parsed received_timestamp is invalid")

            interval_start = self._floor_timestamp(
                timestamp=received_timestamp,
                interval_minutes=interval_minutes,
            )

            grouped_records[
                (
                    instrument_id,
                    interval_start,
                )
            ].append(parsed_record)

        candles = [
            self._build_candle(
                instrument_id=instrument_id,
                interval_start=interval_start,
                records=group,
                interval_minutes=interval_minutes,
            )
            for (
                instrument_id,
                interval_start,
            ), group in grouped_records.items()
        ]

        candles.sort(
            key=lambda candle: (
                candle.interval_start,
                candle.instrument_id,
            )
        )

        return candles

    @staticmethod
    def _floor_timestamp(
        timestamp: datetime,
        interval_minutes: int,
    ) -> datetime:
        utc_timestamp = timestamp.astimezone(UTC)

        total_minutes = utc_timestamp.hour * 60 + utc_timestamp.minute

        floored_minutes = total_minutes // interval_minutes * interval_minutes

        floored_hour = floored_minutes // 60
        floored_minute = floored_minutes % 60

        return utc_timestamp.replace(
            hour=floored_hour,
            minute=floored_minute,
            second=0,
            microsecond=0,
        )

    def _build_candle(
        self,
        instrument_id: str,
        interval_start: datetime,
        records: list[dict[str, object]],
        interval_minutes: int,
    ) -> TickerCandle:
        records.sort(
            key=lambda record: (
                record["received_timestamp"],
                record["exchange_timestamp"],
            )
        )

        first_record = records[0]
        last_record = records[-1]

        prices = self._decimal_values(
            records,
            "last_price",
        )

        sizes = self._decimal_values(
            records,
            "last_size",
        )

        spreads = self._decimal_values(
            records,
            "spread",
        )

        spread_sum = sum(
            spreads,
            start=Decimal("0"),
        )

        tick_count = len(records)

        average_spread = spread_sum / Decimal(tick_count)

        return TickerCandle(
            instrument_id=instrument_id,
            interval_start=interval_start,
            interval_end=(interval_start + timedelta(minutes=interval_minutes)),
            open_price=self._get_decimal(
                first_record,
                "last_price",
            ),
            high_price=max(prices),
            low_price=min(prices),
            close_price=self._get_decimal(
                last_record,
                "last_price",
            ),
            first_bid_price=self._get_decimal(
                first_record,
                "best_bid_price",
            ),
            last_bid_price=self._get_decimal(
                last_record,
                "best_bid_price",
            ),
            first_ask_price=self._get_decimal(
                first_record,
                "best_ask_price",
            ),
            last_ask_price=self._get_decimal(
                last_record,
                "best_ask_price",
            ),
            minimum_spread=min(spreads),
            maximum_spread=max(spreads),
            average_spread=average_spread,
            observed_size=sum(
                sizes,
                start=Decimal("0"),
            ),
            tick_count=tick_count,
            first_exchange_timestamp=(
                self._get_datetime(
                    first_record,
                    "exchange_timestamp",
                )
            ),
            last_exchange_timestamp=(
                self._get_datetime(
                    last_record,
                    "exchange_timestamp",
                )
            ),
            first_received_timestamp=(
                self._get_datetime(
                    first_record,
                    "received_timestamp",
                )
            ),
            last_received_timestamp=(
                self._get_datetime(
                    last_record,
                    "received_timestamp",
                )
            ),
        )

    def _validate_record(
        self,
        record: dict[str, object],
        row_number: int,
    ) -> dict[str, object]:
        instrument_id = record.get("instrument_id")

        if (
            not isinstance(
                instrument_id,
                str,
            )
            or not instrument_id
        ):
            raise CandleAggregationError(f"Invalid instrument_id at row {row_number}")

        decimal_fields = [
            "last_price",
            "last_size",
            "best_bid_price",
            "best_ask_price",
            "spread",
        ]

        for field_name in decimal_fields:
            value = record.get(field_name)

            if not isinstance(
                value,
                Decimal,
            ):
                raise CandleAggregationError(f"Invalid {field_name} at row {row_number}")

            if not value.is_finite():
                raise CandleAggregationError(f"Non-finite {field_name} at row {row_number}")

        last_price = self._get_decimal(
            record,
            "last_price",
        )
        last_size = self._get_decimal(
            record,
            "last_size",
        )
        best_bid_price = self._get_decimal(
            record,
            "best_bid_price",
        )
        best_ask_price = self._get_decimal(
            record,
            "best_ask_price",
        )
        spread = self._get_decimal(
            record,
            "spread",
        )

        if last_price <= 0:
            raise CandleAggregationError(f"last_price must be positive at row {row_number}")

        if last_size < 0:
            raise CandleAggregationError(f"last_size must not be negative at row {row_number}")

        if best_bid_price <= 0:
            raise CandleAggregationError(f"best_bid_price must be positive at row {row_number}")

        if best_ask_price <= 0:
            raise CandleAggregationError(f"best_ask_price must be positive at row {row_number}")

        if best_bid_price > best_ask_price:
            raise CandleAggregationError(f"Crossed market at row {row_number}")

        if spread < 0:
            raise CandleAggregationError(f"spread must not be negative at row {row_number}")

        timestamp_fields = [
            "exchange_timestamp",
            "received_timestamp",
        ]

        for field_name in timestamp_fields:
            value = record.get(field_name)

            if not isinstance(
                value,
                datetime,
            ):
                raise CandleAggregationError(f"Invalid {field_name} at row {row_number}")

            if value.tzinfo is None:
                raise CandleAggregationError(
                    f"{field_name} must contain a timezone at row {row_number}"
                )

        return record

    @staticmethod
    def _decimal_values(
        records: Iterable[dict[str, object]],
        field_name: str,
    ) -> list[Decimal]:
        return [
            TickerCandleAggregator._get_decimal(
                record,
                field_name,
            )
            for record in records
        ]

    @staticmethod
    def _get_decimal(
        record: dict[str, object],
        field_name: str,
    ) -> Decimal:
        value = record[field_name]

        if not isinstance(
            value,
            Decimal,
        ):
            raise CandleAggregationError(f"{field_name} is not a Decimal")

        return value

    @staticmethod
    def _get_datetime(
        record: dict[str, object],
        field_name: str,
    ) -> datetime:
        value = record[field_name]

        if not isinstance(
            value,
            datetime,
        ):
            raise CandleAggregationError(f"{field_name} is not a datetime")

        return value.astimezone(UTC)

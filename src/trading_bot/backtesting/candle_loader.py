from datetime import datetime
from decimal import Decimal

import pyarrow as pa

from trading_bot.market_data.candle import (
    TickerCandle,
)


class CandleLoadError(RuntimeError):
    """Raised when Arrow candle rows are invalid."""


class CandleTableLoader:
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
    }

    def load(
        self,
        table: pa.Table,
    ) -> list[TickerCandle]:
        missing_columns = self.REQUIRED_COLUMNS - set(table.column_names)

        if missing_columns:
            raise CandleLoadError(
                "Candle table is missing required columns: " + ", ".join(sorted(missing_columns))
            )

        candles: list[TickerCandle] = []

        for row_number, record in enumerate(
            table.to_pylist(),
            start=1,
        ):
            try:
                candle = TickerCandle(
                    instrument_id=self._string(
                        record,
                        "instrument_id",
                    ),
                    interval_start=self._datetime(
                        record,
                        "interval_start",
                    ),
                    interval_end=self._datetime(
                        record,
                        "interval_end",
                    ),
                    open_price=self._decimal(
                        record,
                        "open_price",
                    ),
                    high_price=self._decimal(
                        record,
                        "high_price",
                    ),
                    low_price=self._decimal(
                        record,
                        "low_price",
                    ),
                    close_price=self._decimal(
                        record,
                        "close_price",
                    ),
                    first_bid_price=self._decimal(
                        record,
                        "first_bid_price",
                    ),
                    last_bid_price=self._decimal(
                        record,
                        "last_bid_price",
                    ),
                    first_ask_price=self._decimal(
                        record,
                        "first_ask_price",
                    ),
                    last_ask_price=self._decimal(
                        record,
                        "last_ask_price",
                    ),
                    minimum_spread=self._decimal(
                        record,
                        "minimum_spread",
                    ),
                    maximum_spread=self._decimal(
                        record,
                        "maximum_spread",
                    ),
                    average_spread=self._decimal(
                        record,
                        "average_spread",
                    ),
                    observed_size=self._decimal(
                        record,
                        "observed_size",
                    ),
                    tick_count=self._integer(
                        record,
                        "tick_count",
                    ),
                    first_exchange_timestamp=(
                        self._datetime(
                            record,
                            "first_exchange_timestamp",
                        )
                    ),
                    last_exchange_timestamp=(
                        self._datetime(
                            record,
                            "last_exchange_timestamp",
                        )
                    ),
                    first_received_timestamp=(
                        self._datetime(
                            record,
                            "first_received_timestamp",
                        )
                    ),
                    last_received_timestamp=(
                        self._datetime(
                            record,
                            "last_received_timestamp",
                        )
                    ),
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise CandleLoadError(f"Invalid candle row {row_number}: {error}") from error

            candles.append(candle)

        candles.sort(
            key=lambda candle: (
                candle.interval_start,
                candle.instrument_id,
            )
        )

        return candles

    @staticmethod
    def _string(
        record: dict[str, object],
        field_name: str,
    ) -> str:
        value = record[field_name]

        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        return value

    @staticmethod
    def _decimal(
        record: dict[str, object],
        field_name: str,
    ) -> Decimal:
        value = record[field_name]

        if not isinstance(value, Decimal):
            raise TypeError(f"{field_name} must be a Decimal")

        return value

    @staticmethod
    def _datetime(
        record: dict[str, object],
        field_name: str,
    ) -> datetime:
        value = record[field_name]

        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime")

        if value.tzinfo is None:
            raise ValueError(f"{field_name} must contain a timezone")

        return value

    @staticmethod
    def _integer(
        record: dict[str, object],
        field_name: str,
    ) -> int:
        value = record[field_name]

        if not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer")

        return value

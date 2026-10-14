from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import pyarrow as pa


@dataclass(slots=True)
class CandleValidationIssue:
    code: str
    message: str
    symbol: str | None = None
    interval_start: datetime | None = None


@dataclass(slots=True)
class CandleValidationReport:
    total_records: int
    invalid_records: int
    duplicate_intervals: int
    missing_intervals: int
    session_gaps: int
    issues: list[CandleValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.invalid_records == 0 and self.duplicate_intervals == 0


class CandleValidator:
    REQUIRED_COLUMNS = {
        "instrument_id",
        "interval_start",
        "interval_end",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
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

    def validate(
        self,
        table: pa.Table,
        interval_minutes: int = 1,
    ) -> CandleValidationReport:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")

        missing_columns = self.REQUIRED_COLUMNS - set(table.column_names)

        if missing_columns:
            raise ValueError(
                "Candle table is missing columns: " + ", ".join(sorted(missing_columns))
            )

        issues: list[CandleValidationIssue] = []
        invalid_rows: set[int] = set()

        seen_intervals: set[tuple[str, datetime]] = set()

        duplicate_intervals = 0

        grouped_starts: dict[
            str,
            list[datetime],
        ] = {}

        for row_number, record in enumerate(
            table.to_pylist(),
            start=1,
        ):
            symbol = self._get_string(
                record,
                "instrument_id",
            )

            interval_start = self._get_datetime(
                record,
                "interval_start",
            )

            interval_end = self._get_datetime(
                record,
                "interval_end",
            )

            grouped_starts.setdefault(
                symbol,
                [],
            ).append(interval_start)

            interval_key = (
                symbol,
                interval_start,
            )

            if interval_key in seen_intervals:
                duplicate_intervals += 1

                issues.append(
                    CandleValidationIssue(
                        code="duplicate_interval",
                        message=("Duplicate candle interval"),
                        symbol=symbol,
                        interval_start=interval_start,
                    )
                )
            else:
                seen_intervals.add(interval_key)

            row_errors = self._validate_record(
                record=record,
                symbol=symbol,
                interval_start=interval_start,
                interval_end=interval_end,
                interval_minutes=interval_minutes,
            )

            if row_errors:
                invalid_rows.add(row_number)
                issues.extend(row_errors)

        missing_intervals, session_gaps = self._find_missing_intervals(
            grouped_starts=grouped_starts,
            interval_minutes=interval_minutes,
            maximum_continuous_gap_minutes=5,
            issues=issues,
        )

        return CandleValidationReport(
            total_records=table.num_rows,
            invalid_records=len(invalid_rows),
            duplicate_intervals=duplicate_intervals,
            missing_intervals=missing_intervals,
            session_gaps=session_gaps,
            issues=issues,
        )

    def _validate_record(
        self,
        record: dict[str, object],
        symbol: str,
        interval_start: datetime,
        interval_end: datetime,
        interval_minutes: int,
    ) -> list[CandleValidationIssue]:
        issues: list[CandleValidationIssue] = []

        expected_end = interval_start + timedelta(minutes=interval_minutes)

        if interval_end != expected_end:
            issues.append(
                self._issue(
                    code="invalid_interval_end",
                    message=("interval_end does not match the configured interval"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        open_price = self._get_decimal(
            record,
            "open_price",
        )
        high_price = self._get_decimal(
            record,
            "high_price",
        )
        low_price = self._get_decimal(
            record,
            "low_price",
        )
        close_price = self._get_decimal(
            record,
            "close_price",
        )

        if high_price < low_price:
            issues.append(
                self._issue(
                    code="invalid_price_range",
                    message=("high_price is below low_price"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        if not (low_price <= open_price <= high_price):
            issues.append(
                self._issue(
                    code="invalid_open_price",
                    message=("open_price is outside the candle range"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        if not (low_price <= close_price <= high_price):
            issues.append(
                self._issue(
                    code="invalid_close_price",
                    message=("close_price is outside the candle range"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        minimum_spread = self._get_decimal(
            record,
            "minimum_spread",
        )
        maximum_spread = self._get_decimal(
            record,
            "maximum_spread",
        )
        average_spread = self._get_decimal(
            record,
            "average_spread",
        )

        if minimum_spread < 0:
            issues.append(
                self._issue(
                    code="negative_spread",
                    message=("minimum_spread is negative"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        if maximum_spread < minimum_spread:
            issues.append(
                self._issue(
                    code="invalid_spread_range",
                    message=("maximum_spread is below minimum_spread"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        if not (minimum_spread <= average_spread <= maximum_spread):
            issues.append(
                self._issue(
                    code="invalid_average_spread",
                    message=("average_spread is outside the spread range"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        observed_size = self._get_decimal(
            record,
            "observed_size",
        )

        if observed_size < 0:
            issues.append(
                self._issue(
                    code="negative_observed_size",
                    message=("observed_size is negative"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        tick_count = record.get("tick_count")

        if not isinstance(tick_count, int) or tick_count <= 0:
            issues.append(
                self._issue(
                    code="invalid_tick_count",
                    message=("tick_count must be positive"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        first_exchange = self._get_datetime(
            record,
            "first_exchange_timestamp",
        )
        last_exchange = self._get_datetime(
            record,
            "last_exchange_timestamp",
        )
        first_received = self._get_datetime(
            record,
            "first_received_timestamp",
        )
        last_received = self._get_datetime(
            record,
            "last_received_timestamp",
        )

        if first_exchange > last_exchange:
            issues.append(
                self._issue(
                    code="invalid_exchange_order",
                    message=("first exchange timestamp is after the last timestamp"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        if first_received > last_received:
            issues.append(
                self._issue(
                    code="invalid_received_order",
                    message=("first received timestamp is after the last timestamp"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        if not (interval_start <= first_received < interval_end):
            issues.append(
                self._issue(
                    code="first_received_outside_interval",
                    message=("first received timestamp is outside the candle interval"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        if not (interval_start <= last_received < interval_end):
            issues.append(
                self._issue(
                    code="last_received_outside_interval",
                    message=("last received timestamp is outside the candle interval"),
                    symbol=symbol,
                    interval_start=interval_start,
                )
            )

        return issues

    @staticmethod
    def _find_missing_intervals(
        grouped_starts: dict[
            str,
            list[datetime],
        ],
        interval_minutes: int,
        maximum_continuous_gap_minutes: int,
        issues: list[CandleValidationIssue],
    ) -> tuple[int, int]:
        interval_delta = timedelta(minutes=interval_minutes)

        maximum_continuous_gap = timedelta(minutes=maximum_continuous_gap_minutes)

        missing_count = 0
        session_gap_count = 0

        for symbol, starts in grouped_starts.items():
            unique_starts = sorted(set(starts))

            for previous, current in zip(
                unique_starts,
                unique_starts[1:],
                strict=False,
            ):
                gap = current - previous

                if gap <= interval_delta:
                    continue

                missing_duration = gap - interval_delta

                if missing_duration > maximum_continuous_gap:
                    session_gap_count += 1

                    issues.append(
                        CandleValidationIssue(
                            code="capture_session_gap",
                            message=(f"Large gap between recording sessions: {missing_duration}"),
                            symbol=symbol,
                            interval_start=(previous + interval_delta),
                        )
                    )

                    continue

                expected = previous + interval_delta

                while expected < current:
                    missing_count += 1

                    issues.append(
                        CandleValidationIssue(
                            code="missing_interval",
                            message=("Expected candle interval is missing"),
                            symbol=symbol,
                            interval_start=expected,
                        )
                    )

                    expected += interval_delta

        return missing_count, session_gap_count

    @staticmethod
    def _issue(
        code: str,
        message: str,
        symbol: str,
        interval_start: datetime,
    ) -> CandleValidationIssue:
        return CandleValidationIssue(
            code=code,
            message=message,
            symbol=symbol,
            interval_start=interval_start,
        )

    @staticmethod
    def _get_string(
        record: dict[str, object],
        field_name: str,
    ) -> str:
        value = record.get(field_name)

        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")

        return value

    @staticmethod
    def _get_decimal(
        record: dict[str, object],
        field_name: str,
    ) -> Decimal:
        value = record.get(field_name)

        if not isinstance(value, Decimal):
            raise ValueError(f"{field_name} must be a Decimal")

        return value

    @staticmethod
    def _get_datetime(
        record: dict[str, object],
        field_name: str,
    ) -> datetime:
        value = record.get(field_name)

        if not isinstance(value, datetime):
            raise ValueError(f"{field_name} must be a datetime")

        if value.tzinfo is None:
            raise ValueError(f"{field_name} must contain a timezone")

        return value

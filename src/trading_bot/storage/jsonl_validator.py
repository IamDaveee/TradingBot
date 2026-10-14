import json
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ValidationIssue(BaseModel):
    line_number: int
    issue_type: str
    message: str


class ValidationReport(BaseModel):
    file_path: Path
    total_lines: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    symbol_counts: dict[str, int]
    issues: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return self.invalid_records == 0


class JsonlTickerValidator:
    REQUIRED_FIELDS = {
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

    def __init__(
        self,
        allowed_symbols: set[str],
        maximum_clock_difference_seconds: float = 2.0,
    ) -> None:
        if not allowed_symbols:
            raise ValueError("At least one allowed symbol is required")

        if maximum_clock_difference_seconds <= 0:
            raise ValueError("Maximum clock difference must be positive")

        self._allowed_symbols = allowed_symbols
        self._maximum_clock_difference_seconds = maximum_clock_difference_seconds

    def validate_file(
        self,
        file_path: Path,
    ) -> ValidationReport:
        if not file_path.exists():
            raise FileNotFoundError(f"Ticker file does not exist: {file_path}")

        issues: list[ValidationIssue] = []
        symbol_counts: Counter[str] = Counter()
        seen_records: set[tuple[str, str, str, str]] = set()

        total_lines = 0
        valid_records = 0
        duplicate_records = 0

        with file_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                total_lines += 1
                stripped_line = line.strip()

                if not stripped_line:
                    issues.append(
                        ValidationIssue(
                            line_number=line_number,
                            issue_type="empty_line",
                            message="Line is empty",
                        )
                    )
                    continue

                try:
                    payload: Any = json.loads(stripped_line)
                except json.JSONDecodeError as error:
                    issues.append(
                        ValidationIssue(
                            line_number=line_number,
                            issue_type="invalid_json",
                            message=str(error),
                        )
                    )
                    continue

                if not isinstance(payload, dict):
                    issues.append(
                        ValidationIssue(
                            line_number=line_number,
                            issue_type="invalid_record_type",
                            message=("Record must be a JSON object"),
                        )
                    )
                    continue

                record_issues = self._validate_record(
                    payload=payload,
                    line_number=line_number,
                )

                issues.extend(record_issues)

                if record_issues:
                    continue

                symbol = str(payload["instrument_id"])
                symbol_counts[symbol] += 1

                record_identity = (
                    symbol,
                    str(payload["exchange_timestamp"]),
                    str(payload["best_bid_price"]),
                    str(payload["best_ask_price"]),
                )

                if record_identity in seen_records:
                    duplicate_records += 1
                else:
                    seen_records.add(record_identity)

                valid_records += 1

        invalid_records = total_lines - valid_records

        return ValidationReport(
            file_path=file_path,
            total_lines=total_lines,
            valid_records=valid_records,
            invalid_records=invalid_records,
            duplicate_records=duplicate_records,
            symbol_counts=dict(symbol_counts),
            issues=issues,
        )

    def _validate_record(
        self,
        payload: dict[str, Any],
        line_number: int,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        missing_fields = self.REQUIRED_FIELDS - payload.keys()

        if missing_fields:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="missing_fields",
                    message=("Missing fields: " + ", ".join(sorted(missing_fields))),
                )
            )
            return issues

        symbol = str(payload["instrument_id"])

        if symbol not in self._allowed_symbols:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="unexpected_symbol",
                    message=f"Unexpected symbol: {symbol}",
                )
            )

        try:
            last_price = Decimal(str(payload["last_price"]))
            last_size = Decimal(str(payload["last_size"]))
            ask_price = Decimal(str(payload["best_ask_price"]))
            ask_size = Decimal(str(payload["best_ask_size"]))
            bid_price = Decimal(str(payload["best_bid_price"]))
            bid_size = Decimal(str(payload["best_bid_size"]))
        except InvalidOperation:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="invalid_decimal",
                    message="A numeric field is invalid",
                )
            )
            return issues

        if last_price <= 0 or ask_price <= 0 or bid_price <= 0:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="invalid_price",
                    message="Prices must be positive",
                )
            )

        if last_size < 0 or ask_size < 0 or bid_size < 0:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="invalid_size",
                    message="Sizes must not be negative",
                )
            )

        if ask_price < bid_price:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="crossed_market",
                    message=(f"Ask {ask_price} is below bid {bid_price}"),
                )
            )

        try:
            exchange_timestamp = datetime.fromisoformat(str(payload["exchange_timestamp"]))
            received_timestamp = datetime.fromisoformat(str(payload["received_timestamp"]))
            written_timestamp = datetime.fromisoformat(str(payload["written_timestamp"]))
        except ValueError as error:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="invalid_timestamp",
                    message=str(error),
                )
            )
            return issues

        clock_difference = abs((received_timestamp - exchange_timestamp).total_seconds())

        if clock_difference > self._maximum_clock_difference_seconds:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="clock_difference",
                    message=(f"Exchange and receipt timestamps differ by {clock_difference:.6f}s"),
                )
            )

        if written_timestamp < received_timestamp:
            issues.append(
                ValidationIssue(
                    line_number=line_number,
                    issue_type="invalid_write_timestamp",
                    message=("Written timestamp precedes receipt timestamp"),
                )
            )

        return issues

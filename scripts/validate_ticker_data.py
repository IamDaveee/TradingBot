import argparse
from pathlib import Path

from trading_bot.settings import load_settings
from trading_bot.storage.jsonl_validator import (
    JsonlTickerValidator,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stored ticker JSONL data")

    parser.add_argument(
        "file",
        type=Path,
        help="Path to a ticker JSONL file",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _, config = load_settings()

    validator = JsonlTickerValidator(
        allowed_symbols=set(config.market_data.symbols),
    )

    report = validator.validate_file(arguments.file)

    print(f"File: {report.file_path}")
    print(f"Total lines: {report.total_lines}")
    print(f"Valid records: {report.valid_records}")
    print(f"Invalid records: {report.invalid_records}")
    print(f"Duplicate records: {report.duplicate_records}")
    print(f"Symbols: {report.symbol_counts}")

    if report.issues:
        print("\nIssues:")

        for issue in report.issues[:20]:
            print(f"Line {issue.line_number}: {issue.issue_type} — {issue.message}")

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

from trading_bot.storage.jsonl_validator import (
    JsonlTickerValidator,
)


def create_valid_record() -> dict[str, str]:
    return {
        "instrument_id": "BTC-USDT",
        "last_price": "60000.1",
        "last_size": "0.01",
        "best_ask_price": "60000.2",
        "best_ask_size": "1.2",
        "best_bid_price": "60000.1",
        "best_bid_size": "1.1",
        "spread": "0.1",
        "spread_percentage": "0.00016",
        "exchange_timestamp": ("2026-06-27T14:20:25+00:00"),
        "received_timestamp": ("2026-06-27T14:20:25.1+00:00"),
        "written_timestamp": ("2026-06-27T14:20:25.2+00:00"),
    }


def test_validator_accepts_valid_record(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "tickers.jsonl"

    file_path.write_text(
        json.dumps(create_valid_record()) + "\n",
        encoding="utf-8",
    )

    validator = JsonlTickerValidator(
        allowed_symbols={"BTC-USDT"},
    )

    report = validator.validate_file(file_path)

    assert report.is_valid is True
    assert report.total_lines == 1
    assert report.valid_records == 1
    assert report.invalid_records == 0
    assert report.symbol_counts == {"BTC-USDT": 1}


def test_validator_detects_invalid_json(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "tickers.jsonl"
    file_path.write_text(
        "{not valid json}\n",
        encoding="utf-8",
    )

    validator = JsonlTickerValidator(
        allowed_symbols={"BTC-USDT"},
    )

    report = validator.validate_file(file_path)

    assert report.is_valid is False
    assert report.invalid_records == 1
    assert report.issues[0].issue_type == ("invalid_json")


def test_validator_detects_crossed_market(
    tmp_path: Path,
) -> None:
    record = create_valid_record()
    record["best_ask_price"] = "59999"
    record["best_bid_price"] = "60000"

    file_path = tmp_path / "tickers.jsonl"
    file_path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    validator = JsonlTickerValidator(
        allowed_symbols={"BTC-USDT"},
    )

    report = validator.validate_file(file_path)

    assert report.is_valid is False
    assert any(issue.issue_type == "crossed_market" for issue in report.issues)


def test_validator_counts_duplicates(
    tmp_path: Path,
) -> None:
    line = json.dumps(create_valid_record())

    file_path = tmp_path / "tickers.jsonl"
    file_path.write_text(
        f"{line}\n{line}\n",
        encoding="utf-8",
    )

    validator = JsonlTickerValidator(
        allowed_symbols={"BTC-USDT"},
    )

    report = validator.validate_file(file_path)

    assert report.valid_records == 2
    assert report.duplicate_records == 1

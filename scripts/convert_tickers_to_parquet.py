import argparse
from pathlib import Path

from trading_bot.settings import load_settings
from trading_bot.storage.parquet_converter import (
    TickerParquetConverter,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Convert ticker JSONL data into partitioned Parquet files")
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Ticker JSONL file to convert",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Parquet outputs",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _, config = load_settings()

    converter = TickerParquetConverter(
        output_directory=(config.storage.processed_directory),
        allowed_symbols=set(config.market_data.symbols),
    )

    written_files = converter.convert_file(
        input_file=arguments.input_file,
        overwrite=arguments.overwrite,
    )

    print(f"Created {len(written_files)} Parquet file(s):")

    for file_path in written_files:
        print(f"  {file_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

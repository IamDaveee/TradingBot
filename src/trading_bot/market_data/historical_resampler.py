from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trading_bot.market_data.historical_candle import (
    HistoricalCandle,
)

SUPPORTED_TARGET_INTERVALS: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
}


@dataclass(frozen=True, slots=True)
class HistoricalResampleReport:
    source_interval: str
    target_interval: str
    source_candle_count: int
    complete_source_candle_count: int
    incomplete_source_candle_count: int
    duplicate_source_candle_count: int
    candidate_bucket_count: int
    written_candle_count: int
    skipped_incomplete_bucket_count: int
    skipped_partial_bucket_count: int

    @property
    def skipped_bucket_count(self) -> int:
        return self.skipped_incomplete_bucket_count + self.skipped_partial_bucket_count


@dataclass(frozen=True, slots=True)
class HistoricalResampleResult:
    candles: list[HistoricalCandle]
    report: HistoricalResampleReport


class HistoricalCandleResampler:
    def __init__(
        self,
        source_interval: str = "1m",
    ) -> None:
        if source_interval != "1m":
            raise ValueError("The resampler currently supports only 1m source candles")

        self._source_interval = source_interval
        self._source_interval_minutes = 1

    def resample(
        self,
        candles: list[HistoricalCandle],
        target_interval: str,
    ) -> HistoricalResampleResult:
        if target_interval not in SUPPORTED_TARGET_INTERVALS:
            supported = ", ".join(sorted(SUPPORTED_TARGET_INTERVALS))

            raise ValueError(
                f"Unsupported target interval: {target_interval}. Supported intervals: {supported}"
            )

        target_minutes = SUPPORTED_TARGET_INTERVALS[target_interval]

        if not candles:
            return HistoricalResampleResult(
                candles=[],
                report=HistoricalResampleReport(
                    source_interval=self._source_interval,
                    target_interval=target_interval,
                    source_candle_count=0,
                    complete_source_candle_count=0,
                    incomplete_source_candle_count=0,
                    duplicate_source_candle_count=0,
                    candidate_bucket_count=0,
                    written_candle_count=0,
                    skipped_incomplete_bucket_count=0,
                    skipped_partial_bucket_count=0,
                ),
            )

        self._validate_candles(candles)

        unique_by_key: dict[
            tuple[str, datetime],
            HistoricalCandle,
        ] = {}

        duplicate_count = 0

        for candle in sorted(
            candles,
            key=lambda item: (
                item.instrument_id,
                item.interval_start,
            ),
        ):
            key = (
                candle.instrument_id,
                candle.interval_start,
            )

            if key in unique_by_key:
                duplicate_count += 1

            # Deterministic behavior: the last duplicate wins.
            unique_by_key[key] = candle

        buckets: dict[
            tuple[str, datetime],
            list[HistoricalCandle],
        ] = {}

        for candle in unique_by_key.values():
            bucket_start = self._bucket_start(
                candle.interval_start,
                target_minutes,
            )

            key = (
                candle.instrument_id,
                bucket_start,
            )

            buckets.setdefault(
                key,
                [],
            ).append(candle)

        output: list[HistoricalCandle] = []

        skipped_incomplete = 0
        skipped_partial = 0

        for (
            instrument_id,
            bucket_start,
        ), bucket_candles in sorted(
            buckets.items(),
            key=lambda item: (
                item[0][1],
                item[0][0],
            ),
        ):
            bucket_candles.sort(key=lambda candle: candle.interval_start)

            if any(not candle.is_complete for candle in bucket_candles):
                skipped_incomplete += 1
                continue

            if not self._is_complete_bucket(
                candles=bucket_candles,
                bucket_start=bucket_start,
                target_minutes=target_minutes,
            ):
                skipped_partial += 1
                continue

            first = bucket_candles[0]
            last = bucket_candles[-1]

            output.append(
                HistoricalCandle(
                    instrument_id=instrument_id,
                    interval=target_interval,
                    interval_start=bucket_start,
                    interval_end=(bucket_start + timedelta(minutes=target_minutes)),
                    open_price=first.open_price,
                    high_price=max(candle.high_price for candle in bucket_candles),
                    low_price=min(candle.low_price for candle in bucket_candles),
                    close_price=last.close_price,
                    base_volume=sum(
                        (candle.base_volume for candle in bucket_candles),
                        start=first.base_volume * 0,
                    ),
                    currency_volume=sum(
                        (candle.currency_volume for candle in bucket_candles),
                        start=first.currency_volume * 0,
                    ),
                    quote_volume=sum(
                        (candle.quote_volume for candle in bucket_candles),
                        start=first.quote_volume * 0,
                    ),
                    is_complete=True,
                )
            )

        complete_source_count = sum(1 for candle in unique_by_key.values() if candle.is_complete)

        incomplete_source_count = len(unique_by_key) - complete_source_count

        report = HistoricalResampleReport(
            source_interval=self._source_interval,
            target_interval=target_interval,
            source_candle_count=len(candles),
            complete_source_candle_count=(complete_source_count),
            incomplete_source_candle_count=(incomplete_source_count),
            duplicate_source_candle_count=(duplicate_count),
            candidate_bucket_count=len(buckets),
            written_candle_count=len(output),
            skipped_incomplete_bucket_count=(skipped_incomplete),
            skipped_partial_bucket_count=(skipped_partial),
        )

        return HistoricalResampleResult(
            candles=output,
            report=report,
        )

    def _validate_candles(
        self,
        candles: list[HistoricalCandle],
    ) -> None:
        for candle in candles:
            if candle.interval != self._source_interval:
                raise ValueError(
                    "All source candles must use "
                    f"interval={self._source_interval}; "
                    f"received {candle.interval}"
                )

            if candle.interval_start.tzinfo is None:
                raise ValueError("interval_start must be timezone-aware")

            if candle.interval_end.tzinfo is None:
                raise ValueError("interval_end must be timezone-aware")

            expected_end = candle.interval_start + timedelta(
                minutes=(self._source_interval_minutes)
            )

            if candle.interval_end != expected_end:
                raise ValueError(
                    "Source candle duration does not match the configured source interval"
                )

    @staticmethod
    def _bucket_start(
        value: datetime,
        target_minutes: int,
    ) -> datetime:
        utc_value = value.astimezone(UTC)

        minutes_since_midnight = utc_value.hour * 60 + utc_value.minute

        bucket_minute = minutes_since_midnight // target_minutes * target_minutes

        bucket_hour = bucket_minute // 60
        minute_in_hour = bucket_minute % 60

        return utc_value.replace(
            hour=bucket_hour,
            minute=minute_in_hour,
            second=0,
            microsecond=0,
        )

    @staticmethod
    def _is_complete_bucket(
        candles: list[HistoricalCandle],
        bucket_start: datetime,
        target_minutes: int,
    ) -> bool:
        if len(candles) != target_minutes:
            return False

        expected_starts = [
            bucket_start + timedelta(minutes=offset) for offset in range(target_minutes)
        ]

        actual_starts = [candle.interval_start.astimezone(UTC) for candle in candles]

        return actual_starts == expected_starts

from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_bot.market_data.candle import TickerCandle


@dataclass(frozen=True, slots=True)
class CoverageGap:
    instrument_id: str
    start: datetime
    end: datetime
    missing_intervals: int
    is_session_gap: bool


@dataclass(frozen=True, slots=True)
class SymbolCoverage:
    instrument_id: str
    candle_count: int
    first_interval_start: datetime
    last_interval_start: datetime
    expected_interval_count: int
    missing_interval_count: int
    session_gap_count: int
    duplicate_interval_count: int

    @property
    def coverage_percentage(self) -> float:
        if self.expected_interval_count <= 0:
            return 100.0

        present_intervals = self.expected_interval_count - self.missing_interval_count

        return present_intervals / self.expected_interval_count * 100.0


@dataclass(frozen=True, slots=True)
class CoverageReport:
    symbols: list[SymbolCoverage]
    gaps: list[CoverageGap]

    @property
    def total_candles(self) -> int:
        return sum(symbol.candle_count for symbol in self.symbols)

    @property
    def total_missing_intervals(self) -> int:
        return sum(symbol.missing_interval_count for symbol in self.symbols)

    @property
    def total_session_gaps(self) -> int:
        return sum(symbol.session_gap_count for symbol in self.symbols)

    @property
    def total_duplicates(self) -> int:
        return sum(symbol.duplicate_interval_count for symbol in self.symbols)


class CandleCoverageAnalyzer:
    def __init__(
        self,
        interval_minutes: int = 1,
        session_gap_threshold_minutes: int = 5,
    ) -> None:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")

        if session_gap_threshold_minutes < 0:
            raise ValueError("session_gap_threshold_minutes must not be negative")

        self._interval = timedelta(minutes=interval_minutes)

        self._session_gap_threshold = timedelta(minutes=session_gap_threshold_minutes)

    def analyze(
        self,
        candles: list[TickerCandle],
    ) -> CoverageReport:
        grouped: dict[
            str,
            list[datetime],
        ] = {}

        for candle in candles:
            grouped.setdefault(
                candle.instrument_id,
                [],
            ).append(candle.interval_start)

        symbol_reports: list[SymbolCoverage] = []
        gaps: list[CoverageGap] = []

        for instrument_id in sorted(grouped):
            starts = sorted(grouped[instrument_id])

            if not starts:
                continue

            duplicate_count = len(starts) - len(set(starts))

            unique_starts = sorted(set(starts))

            missing_interval_count = 0
            session_gap_count = 0

            for previous, current in zip(
                unique_starts,
                unique_starts[1:],
                strict=False,
            ):
                difference = current - previous

                if difference <= self._interval:
                    continue

                missing_duration = difference - self._interval

                missing_intervals = int(missing_duration / self._interval)

                is_session_gap = missing_duration > self._session_gap_threshold

                if is_session_gap:
                    session_gap_count += 1
                else:
                    missing_interval_count += missing_intervals

                gaps.append(
                    CoverageGap(
                        instrument_id=instrument_id,
                        start=(previous + self._interval),
                        end=current,
                        missing_intervals=(missing_intervals),
                        is_session_gap=(is_session_gap),
                    )
                )

            total_span = unique_starts[-1] - unique_starts[0]

            expected_interval_count = int(total_span / self._interval) + 1

            symbol_reports.append(
                SymbolCoverage(
                    instrument_id=instrument_id,
                    candle_count=len(unique_starts),
                    first_interval_start=(unique_starts[0]),
                    last_interval_start=(unique_starts[-1]),
                    expected_interval_count=(expected_interval_count),
                    missing_interval_count=(missing_interval_count),
                    session_gap_count=(session_gap_count),
                    duplicate_interval_count=(duplicate_count),
                )
            )

        return CoverageReport(
            symbols=symbol_reports,
            gaps=gaps,
        )


def validate_minimum_history(
    candles: list[TickerCandle],
    required_candles: int,
) -> None:
    if required_candles <= 0:
        raise ValueError("required_candles must be greater than zero")

    grouped_counts: dict[str, int] = {}

    for candle in candles:
        grouped_counts[candle.instrument_id] = (
            grouped_counts.get(
                candle.instrument_id,
                0,
            )
            + 1
        )

    if not grouped_counts:
        raise ValueError("No candle data is available")

    insufficient = {
        symbol: count for symbol, count in grouped_counts.items() if count < required_candles
    }

    if insufficient:
        descriptions = ", ".join(
            f"{symbol}={count}" for symbol, count in sorted(insufficient.items())
        )

        raise ValueError(
            "Insufficient candle history. "
            f"Required at least {required_candles}; "
            f"available: {descriptions}"
        )

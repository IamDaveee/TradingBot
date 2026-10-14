from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from itertools import groupby

from trading_bot.backtesting.fees import (
    PercentageFeeModel,
)
from trading_bot.backtesting.models import (
    Fill,
    OrderRequest,
    OrderSide,
)
from trading_bot.backtesting.portfolio import (
    Portfolio,
    PortfolioError,
)
from trading_bot.backtesting.strategy import (
    Strategy,
    StrategyContext,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


@dataclass(frozen=True, slots=True)
class RejectedOrder:
    order: OrderRequest
    reason: str
    rejected_at: datetime


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    timestamp: datetime
    cash: Decimal
    equity: Decimal
    realized_pnl: Decimal
    fees_paid: Decimal


@dataclass(slots=True)
class BacktestResult:
    fills: list[Fill] = field(default_factory=list)
    rejected_orders: list[RejectedOrder] = field(default_factory=list)
    equity_curve: list[EquitySnapshot] = field(default_factory=list)
    unexecuted_orders: list[OrderRequest] = field(default_factory=list)

    @property
    def final_equity(self) -> Decimal | None:
        if not self.equity_curve:
            return None

        return self.equity_curve[-1].equity


class BacktestExecutionEngine:
    def __init__(
        self,
        portfolio: Portfolio,
        fee_model: PercentageFeeModel,
        slippage_basis_points: Decimal = Decimal("0"),
        liquidate_at_end: bool = False,
    ) -> None:
        if not slippage_basis_points.is_finite():
            raise ValueError("slippage_basis_points must be finite")

        if slippage_basis_points < 0:
            raise ValueError("slippage_basis_points must not be negative")

        if slippage_basis_points >= Decimal("10000"):
            raise ValueError("slippage_basis_points must be below 10000")

        self._portfolio = portfolio
        self._fee_model = fee_model
        self._slippage_basis_points = slippage_basis_points
        self._liquidate_at_end = liquidate_at_end

    def run(
        self,
        candles: Iterable[TickerCandle],
        orders: Iterable[OrderRequest],
    ) -> BacktestResult:
        sorted_candles = sorted(
            candles,
            key=lambda candle: (
                candle.interval_start,
                candle.instrument_id,
            ),
        )

        sorted_orders = sorted(
            orders,
            key=lambda order: (
                order.created_at,
                order.instrument_id,
                order.side.value,
            ),
        )

        if not sorted_candles:
            return BacktestResult(unexecuted_orders=sorted_orders)

        self._validate_candle_order(sorted_candles)

        result = BacktestResult()

        pending_orders: list[OrderRequest] = []
        order_index = 0

        latest_mark_prices: dict[
            str,
            Decimal,
        ] = {}

        last_candles: dict[
            str,
            TickerCandle,
        ] = {}

        grouped_candles = groupby(
            sorted_candles,
            key=lambda candle: candle.interval_start,
        )

        for (
            interval_start,
            candle_group,
        ) in grouped_candles:
            current_candles = list(candle_group)

            while (
                order_index < len(sorted_orders)
                and sorted_orders[order_index].created_at <= interval_start
            ):
                pending_orders.append(sorted_orders[order_index])
                order_index += 1

            candles_by_symbol = {candle.instrument_id: candle for candle in current_candles}

            pending_orders = self._execute_pending_orders(
                pending_orders=(pending_orders),
                candles_by_symbol=(candles_by_symbol),
                result=result,
            )

            for candle in current_candles:
                latest_mark_prices[candle.instrument_id] = candle.close_price

                last_candles[candle.instrument_id] = candle

            snapshot_time = max(candle.interval_end for candle in current_candles)

            self._append_equity_snapshot(
                result=result,
                timestamp=snapshot_time,
                mark_prices=latest_mark_prices,
            )

        pending_orders.extend(sorted_orders[order_index:])

        if self._liquidate_at_end:
            self._liquidate_positions(
                last_candles=last_candles,
                mark_prices=latest_mark_prices,
                result=result,
            )

        result.unexecuted_orders = pending_orders

        return result

    def run_strategy(
        self,
        candles: Iterable[TickerCandle],
        strategy: Strategy,
    ) -> BacktestResult:
        sorted_candles = sorted(
            candles,
            key=lambda candle: (
                candle.interval_start,
                candle.instrument_id,
            ),
        )

        if not sorted_candles:
            return BacktestResult()

        self._validate_candle_order(sorted_candles)

        result = BacktestResult()

        pending_orders: list[OrderRequest] = []

        latest_mark_prices: dict[
            str,
            Decimal,
        ] = {}

        last_candles: dict[
            str,
            TickerCandle,
        ] = {}

        grouped_candles = groupby(
            sorted_candles,
            key=lambda candle: candle.interval_start,
        )

        for (
            interval_start,
            candle_group,
        ) in grouped_candles:
            current_candles = list(candle_group)

            candles_by_symbol = {candle.instrument_id: candle for candle in current_candles}

            eligible_orders: list[OrderRequest] = []

            future_orders: list[OrderRequest] = []

            for order in pending_orders:
                if order.created_at <= interval_start:
                    eligible_orders.append(order)
                else:
                    future_orders.append(order)

            unfilled_eligible_orders = self._execute_pending_orders(
                pending_orders=eligible_orders,
                candles_by_symbol=(candles_by_symbol),
                result=result,
            )

            pending_orders = [
                *future_orders,
                *unfilled_eligible_orders,
            ]

            for candle in current_candles:
                latest_mark_prices[candle.instrument_id] = candle.close_price

                last_candles[candle.instrument_id] = candle

            for candle in current_candles:
                context = StrategyContext(
                    cash=self._portfolio.cash,
                    equity=self._portfolio.equity(latest_mark_prices),
                    position=(self._portfolio.get_position(candle.instrument_id)),
                )

                generated_orders = strategy.on_candle(
                    candle=candle,
                    context=context,
                )

                for order in generated_orders:
                    self._validate_strategy_order(
                        order=order,
                        source_candle=candle,
                    )

                    pending_orders.append(order)

            snapshot_time = max(candle.interval_end for candle in current_candles)

            self._append_equity_snapshot(
                result=result,
                timestamp=snapshot_time,
                mark_prices=latest_mark_prices,
            )

        if self._liquidate_at_end:
            self._liquidate_positions(
                last_candles=last_candles,
                mark_prices=latest_mark_prices,
                result=result,
            )

        result.unexecuted_orders = pending_orders

        return result

    def _execute_pending_orders(
        self,
        pending_orders: list[OrderRequest],
        candles_by_symbol: dict[
            str,
            TickerCandle,
        ],
        result: BacktestResult,
    ) -> list[OrderRequest]:
        remaining_orders: list[OrderRequest] = []

        for order in pending_orders:
            candle = candles_by_symbol.get(order.instrument_id)

            if candle is None:
                remaining_orders.append(order)
                continue

            fill = self._create_fill(
                order=order,
                base_price=candle.open_price,
                timestamp=(candle.interval_start),
            )

            try:
                self._portfolio.apply_fill(fill)
            except PortfolioError as error:
                result.rejected_orders.append(
                    RejectedOrder(
                        order=order,
                        reason=str(error),
                        rejected_at=(candle.interval_start),
                    )
                )
                continue

            result.fills.append(fill)

        return remaining_orders

    def _create_fill(
        self,
        order: OrderRequest,
        base_price: Decimal,
        timestamp: datetime,
    ) -> Fill:
        execution_price = self._apply_slippage(
            side=order.side,
            base_price=base_price,
        )

        fee = self._fee_model.calculate(
            quantity=order.quantity,
            price=execution_price,
        )

        return Fill(
            instrument_id=(order.instrument_id),
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            fee=fee,
            timestamp=timestamp,
        )

    def _apply_slippage(
        self,
        side: OrderSide,
        base_price: Decimal,
    ) -> Decimal:
        if not base_price.is_finite():
            raise ValueError("base_price must be finite")

        if base_price <= 0:
            raise ValueError("base_price must be positive")

        slippage_fraction = self._slippage_basis_points / Decimal("10000")

        if side == OrderSide.BUY:
            return base_price * (Decimal("1") + slippage_fraction)

        return base_price * (Decimal("1") - slippage_fraction)

    def _append_equity_snapshot(
        self,
        result: BacktestResult,
        timestamp: datetime,
        mark_prices: dict[str, Decimal],
    ) -> None:
        equity = self._portfolio.equity(mark_prices)

        result.equity_curve.append(
            EquitySnapshot(
                timestamp=timestamp,
                cash=self._portfolio.cash,
                equity=equity,
                realized_pnl=(self._portfolio.total_realized_pnl),
                fees_paid=(self._portfolio.fees_paid),
            )
        )

    def _liquidate_positions(
        self,
        last_candles: dict[
            str,
            TickerCandle,
        ],
        mark_prices: dict[str, Decimal],
        result: BacktestResult,
    ) -> None:
        liquidation_timestamps: list[datetime] = []

        for symbol, candle in sorted(last_candles.items()):
            position = self._portfolio.get_position(symbol)

            if not position.is_open:
                continue

            order = OrderRequest(
                instrument_id=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                created_at=(candle.interval_end),
            )

            fill = self._create_fill(
                order=order,
                base_price=candle.close_price,
                timestamp=(candle.interval_end),
            )

            try:
                self._portfolio.apply_fill(fill)
            except PortfolioError as error:
                result.rejected_orders.append(
                    RejectedOrder(
                        order=order,
                        reason=str(error),
                        rejected_at=(candle.interval_end),
                    )
                )
                continue

            result.fills.append(fill)

            liquidation_timestamps.append(candle.interval_end)

            mark_prices[symbol] = candle.close_price

        if liquidation_timestamps:
            self._append_equity_snapshot(
                result=result,
                timestamp=max(liquidation_timestamps),
                mark_prices=mark_prices,
            )

    @staticmethod
    def _validate_strategy_order(
        order: OrderRequest,
        source_candle: TickerCandle,
    ) -> None:
        if order.instrument_id != source_candle.instrument_id:
            raise ValueError(
                "Strategy order instrument does not "
                "match the source candle: "
                f"order={order.instrument_id}, "
                f"candle={source_candle.instrument_id}"
            )

        if order.created_at < source_candle.interval_end:
            raise ValueError(
                "Strategy orders must not be created before the source candle has closed"
            )

    @staticmethod
    def _validate_candle_order(
        candles: list[TickerCandle],
    ) -> None:
        seen_keys: set[tuple[str, datetime]] = set()

        last_start_by_symbol: dict[
            str,
            datetime,
        ] = {}

        for candle in candles:
            key = (
                candle.instrument_id,
                candle.interval_start,
            )

            if key in seen_keys:
                raise ValueError(
                    "Duplicate candle supplied to "
                    "execution engine: "
                    f"{candle.instrument_id} "
                    f"{candle.interval_start}"
                )

            seen_keys.add(key)

            previous_start = last_start_by_symbol.get(candle.instrument_id)

            if previous_start is not None and candle.interval_start <= previous_start:
                raise ValueError("Candles must progress forward for each symbol")

            last_start_by_symbol[candle.instrument_id] = candle.interval_start

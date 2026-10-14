from dataclasses import dataclass
from decimal import Decimal

from trading_bot.market_data.candle import TickerCandle


@dataclass(frozen=True, slots=True)
class BuyAndHoldResult:
    instrument_id: str
    initial_cash: Decimal
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    final_equity: Decimal
    absolute_return: Decimal
    return_percentage: Decimal


def calculate_buy_and_hold(
    candles: list[TickerCandle],
    initial_cash: Decimal,
    fee_rate: Decimal = Decimal("0"),
    slippage_basis_points: Decimal = Decimal("0"),
) -> BuyAndHoldResult:
    if not candles:
        raise ValueError("At least one candle is required")

    if initial_cash <= 0:
        raise ValueError("initial_cash must be greater than zero")

    if not initial_cash.is_finite():
        raise ValueError("initial_cash must be finite")

    if fee_rate < 0 or fee_rate >= 1:
        raise ValueError("fee_rate must be within [0, 1)")

    if not fee_rate.is_finite():
        raise ValueError("fee_rate must be finite")

    if slippage_basis_points < 0 or slippage_basis_points >= Decimal("10000"):
        raise ValueError("slippage_basis_points must be within [0, 10000)")

    if not slippage_basis_points.is_finite():
        raise ValueError("slippage_basis_points must be finite")

    symbols = {candle.instrument_id for candle in candles}

    if len(symbols) != 1:
        raise ValueError("Buy-and-hold requires candles for exactly one instrument")

    sorted_candles = sorted(
        candles,
        key=lambda candle: candle.interval_start,
    )

    first_candle = sorted_candles[0]
    last_candle = sorted_candles[-1]

    slippage_fraction = slippage_basis_points / Decimal("10000")

    entry_price = first_candle.open_price * (Decimal("1") + slippage_fraction)

    exit_price = last_candle.close_price * (Decimal("1") - slippage_fraction)

    quantity = initial_cash / (entry_price * (Decimal("1") + fee_rate))

    entry_notional = quantity * entry_price

    entry_fee = entry_notional * fee_rate

    exit_notional = quantity * exit_price

    exit_fee = exit_notional * fee_rate

    final_equity = exit_notional - exit_fee

    absolute_return = final_equity - initial_cash

    return_percentage = absolute_return / initial_cash * Decimal("100")

    return BuyAndHoldResult(
        instrument_id=first_candle.instrument_id,
        initial_cash=initial_cash,
        quantity=quantity,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        final_equity=final_equity,
        absolute_return=absolute_return,
        return_percentage=return_percentage,
    )

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator


class OkxInstrument(BaseModel):
    instrument_id: str
    instrument_type: str
    base_currency: str
    quote_currency: str
    state: str

    tick_size: Decimal = Field(gt=0)
    lot_size: Decimal = Field(gt=0)
    minimum_order_size: Decimal = Field(gt=0)

    @field_validator(
        "tick_size",
        "lot_size",
        "minimum_order_size",
        mode="before",
    )
    @classmethod
    def convert_empty_decimal(cls, value: object) -> object:
        if value == "":
            raise ValueError("Decimal instrument values must not be empty")

        return value

    @property
    def is_live(self) -> bool:
        return self.state == "live"


class OkxTicker(BaseModel):
    instrument_id: str
    last_price: Decimal = Field(gt=0)
    last_size: Decimal = Field(ge=0)
    best_ask_price: Decimal = Field(gt=0)
    best_ask_size: Decimal = Field(ge=0)
    best_bid_price: Decimal = Field(gt=0)
    best_bid_size: Decimal = Field(ge=0)
    exchange_timestamp: datetime
    received_timestamp: datetime

    @property
    def spread(self) -> Decimal:
        return self.best_ask_price - self.best_bid_price

    @property
    def mid_price(self) -> Decimal:
        return (self.best_ask_price + self.best_bid_price) / Decimal("2")

    @property
    def spread_percentage(self) -> Decimal:
        return self.spread / self.mid_price * Decimal("100")

    @property
    def data_age_seconds(self) -> float:
        return (self.received_timestamp - self.exchange_timestamp).total_seconds()

    @property
    def clock_offset_detected(self) -> bool:
        return self.data_age_seconds < -0.25

    @classmethod
    def from_okx_payload(
        cls,
        payload: dict[str, object],
        received_timestamp: datetime | None = None,
    ) -> "OkxTicker":
        timestamp_ms = int(str(payload["ts"]))

        if received_timestamp is None:
            received_timestamp = datetime.now(UTC)

        return cls(
            instrument_id=str(payload["instId"]),
            last_price=Decimal(str(payload["last"])),
            last_size=Decimal(str(payload["lastSz"])),
            best_ask_price=Decimal(str(payload["askPx"])),
            best_ask_size=Decimal(str(payload["askSz"])),
            best_bid_price=Decimal(str(payload["bidPx"])),
            best_bid_size=Decimal(str(payload["bidSz"])),
            exchange_timestamp=datetime.fromtimestamp(
                timestamp_ms / 1000,
                tz=UTC,
            ),
            received_timestamp=received_timestamp,
        )

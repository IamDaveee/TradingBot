from decimal import Decimal
from typing import Any

import httpx

from trading_bot.exchanges.okx.models import OkxInstrument


class OkxApiError(RuntimeError):
    """Raised when OKX returns an unsuccessful API response."""


class OkxPublicRestClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)

    async def get_spot_instruments(self) -> list[OkxInstrument]:
        endpoint = "/api/v5/public/instruments"

        params = {
            "instType": "SPOT",
        }

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        ) as client:
            response = await client.get(
                endpoint,
                params=params,
            )

            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        self._validate_response(payload)

        raw_instruments = payload.get("data")

        if not isinstance(raw_instruments, list):
            raise OkxApiError("OKX response data field is not a list")

        instruments: list[OkxInstrument] = []

        for item in raw_instruments:
            if not isinstance(item, dict):
                continue

            instrument = self._parse_instrument(item)
            instruments.append(instrument)

        return instruments

    @staticmethod
    def _validate_response(payload: dict[str, Any]) -> None:
        code = payload.get("code")
        message = payload.get("msg", "")

        if code != "0":
            raise OkxApiError(f"OKX API returned code {code}: {message}")

    @staticmethod
    def _parse_instrument(
        item: dict[str, Any],
    ) -> OkxInstrument:
        return OkxInstrument(
            instrument_id=str(item.get("instId", "")),
            instrument_type=str(item.get("instType", "")),
            base_currency=str(item.get("baseCcy", "")),
            quote_currency=str(item.get("quoteCcy", "")),
            state=str(item.get("state", "")),
            tick_size=Decimal(str(item.get("tickSz", ""))),
            lot_size=Decimal(str(item.get("lotSz", ""))),
            minimum_order_size=Decimal(str(item.get("minSz", ""))),
        )

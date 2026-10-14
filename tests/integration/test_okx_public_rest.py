import pytest

from trading_bot.exchanges.okx.public_rest import (
    OkxPublicRestClient,
)


@pytest.mark.integration
async def test_okx_spot_instruments_can_be_loaded() -> None:
    client = OkxPublicRestClient(
        base_url="https://www.okx.com",
    )

    instruments = await client.get_spot_instruments()

    assert len(instruments) > 0
    assert any(instrument.instrument_type == "SPOT" for instrument in instruments)

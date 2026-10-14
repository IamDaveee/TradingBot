import asyncio
import signal
from contextlib import suppress

import structlog

from trading_bot.exchanges.okx.public_rest import (
    OkxPublicRestClient,
)
from trading_bot.exchanges.okx.public_websocket import (
    OkxPublicWebSocketClient,
)
from trading_bot.logging_config import configure_logging
from trading_bot.market_data.service import MarketDataService
from trading_bot.settings import load_settings
from trading_bot.storage.jsonl_writer import (
    JsonlTickerWriter,
)
from trading_bot.storage.ticker_storage_worker import (
    TickerStorageWorker,
)


def register_shutdown_handlers(
    shutdown_event: asyncio.Event,
) -> None:
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        if not shutdown_event.is_set():
            shutdown_event.set()

    for signal_name in (
        signal.SIGINT,
        signal.SIGTERM,
    ):
        try:
            loop.add_signal_handler(
                signal_name,
                request_shutdown,
            )
        except NotImplementedError:
            signal.signal(
                signal_name,
                lambda *_: loop.call_soon_threadsafe(request_shutdown),
            )


async def wait_for_market_data_ready(
    market_data_service: MarketDataService,
    timeout_seconds: float,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    while not await market_data_service.is_ready():
        if asyncio.get_running_loop().time() >= deadline:
            return False

        await asyncio.sleep(0.1)

    return True


async def monitor_market_data_health(
    market_data_service: MarketDataService,
    shutdown_event: asyncio.Event,
    interval_seconds: float,
    storage_worker: TickerStorageWorker | None,
) -> None:
    logger = structlog.get_logger("market_data_health_monitor")

    while not shutdown_event.is_set():
        stale_symbols = await market_data_service.get_stale_symbols()
        ready = await market_data_service.is_ready()

        storage_queue_size = storage_worker.queue_size if storage_worker is not None else 0

        storage_queue_capacity = storage_worker.queue_capacity if storage_worker is not None else 0

        storage_records_written = (
            storage_worker.records_written if storage_worker is not None else 0
        )

        if ready and not stale_symbols:
            logger.info(
                "market_data_healthy",
                ready=True,
                stale_symbols=[],
                storage_queue_size=storage_queue_size,
                storage_queue_capacity=(storage_queue_capacity),
                storage_records_written=(storage_records_written),
            )
        else:
            logger.warning(
                "market_data_unhealthy",
                ready=ready,
                stale_symbols=stale_symbols,
                missing_symbols=(await market_data_service.get_missing_symbols()),
                storage_queue_size=storage_queue_size,
                storage_queue_capacity=(storage_queue_capacity),
                storage_records_written=(storage_records_written),
            )

        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=interval_seconds,
            )
        except TimeoutError:
            continue


async def stop_background_task(
    task: asyncio.Task[None],
    timeout_seconds: float = 5.0,
) -> None:
    try:
        await asyncio.wait_for(
            task,
            timeout=timeout_seconds,
        )
    except TimeoutError:
        task.cancel()

        with suppress(asyncio.CancelledError):
            await task


async def run_async() -> int:
    environment, config = load_settings()

    configure_logging(
        level=config.logging.level,
        json_output=config.logging.json_output,
    )

    logger = structlog.get_logger("trading_bot")

    logger.info(
        "application_started",
        application=config.application.name,
        environment=environment.environment,
        exchange=config.exchange.name,
        demo_trading=config.exchange.demo_trading,
        symbols=config.market_data.symbols,
        trading_enabled=config.trading.enabled,
        live_trading_enabled=(environment.live_trading_enabled),
    )

    if environment.live_trading_enabled:
        logger.error(
            "unsafe_development_configuration",
            reason=("Live trading must remain disabled during development."),
        )
        return 1

    rest_client = OkxPublicRestClient(
        base_url=config.exchange.rest_base_url,
    )

    logger.info("loading_spot_instruments")

    instruments = await rest_client.get_spot_instruments()

    instruments_by_id = {instrument.instrument_id: instrument for instrument in instruments}

    logger.info(
        "spot_instruments_loaded",
        total_count=len(instruments),
        live_count=sum(instrument.is_live for instrument in instruments),
    )

    for symbol in config.market_data.symbols:
        instrument = instruments_by_id.get(symbol)

        if instrument is None:
            logger.error(
                "configured_symbol_not_found",
                symbol=symbol,
            )
            return 1

        if not instrument.is_live:
            logger.error(
                "configured_symbol_not_live",
                symbol=symbol,
                state=instrument.state,
            )
            return 1

        logger.info(
            "configured_symbol_found",
            symbol=instrument.instrument_id,
            state=instrument.state,
            base_currency=instrument.base_currency,
            quote_currency=instrument.quote_currency,
            tick_size=str(instrument.tick_size),
            lot_size=str(instrument.lot_size),
            minimum_order_size=str(instrument.minimum_order_size),
        )

    websocket_client = OkxPublicWebSocketClient(
        websocket_url=(config.exchange.public_websocket_url),
        initial_reconnect_delay_seconds=(config.market_data.initial_reconnect_delay_seconds),
        maximum_reconnect_delay_seconds=(config.market_data.maximum_reconnect_delay_seconds),
    )

    storage_worker: TickerStorageWorker | None = None
    storage_task: asyncio.Task[None] | None = None

    if config.storage.enabled:
        ticker_writer = JsonlTickerWriter(
            directory=config.storage.directory,
            flush_every_records=(config.storage.flush_every_records),
        )

        storage_worker = TickerStorageWorker(
            writer=ticker_writer,
            queue_max_size=(config.storage.queue_max_size),
        )

        storage_task = asyncio.create_task(
            storage_worker.run(),
            name="ticker-storage-worker",
        )

        await storage_worker.wait_until_started()

    market_data_service = MarketDataService(
        websocket_client=websocket_client,
        symbols=config.market_data.symbols,
        stale_after_seconds=(config.market_data.stale_after_seconds),
        storage_worker=storage_worker,
    )

    shutdown_event = asyncio.Event()
    register_shutdown_handlers(shutdown_event)

    service_task = asyncio.create_task(
        market_data_service.run(),
        name="market-data-service",
    )

    health_task: asyncio.Task[None] | None = None

    try:
        logger.info(
            "waiting_for_market_data",
            symbols=config.market_data.symbols,
        )

        ready = await wait_for_market_data_ready(
            market_data_service,
            timeout_seconds=15.0,
        )

        if not ready:
            logger.error(
                "market_data_startup_timeout",
                missing_symbols=(await market_data_service.get_missing_symbols()),
            )
            return 1

        logger.info(
            "market_data_ready",
            symbols=config.market_data.symbols,
        )

        snapshot = await market_data_service.get_snapshot()

        for symbol in config.market_data.symbols:
            ticker = snapshot[symbol]

            logger.info(
                "latest_ticker",
                symbol=ticker.instrument_id,
                last_price=str(ticker.last_price),
                best_bid=str(ticker.best_bid_price),
                best_ask=str(ticker.best_ask_price),
                spread=str(ticker.spread),
                received_timestamp=(ticker.received_timestamp.isoformat()),
            )

        health_task = asyncio.create_task(
            monitor_market_data_health(
                market_data_service=market_data_service,
                shutdown_event=shutdown_event,
                interval_seconds=(config.market_data.health_check_interval_seconds),
                storage_worker=storage_worker,
            ),
            name="market-data-health-monitor",
        )

        logger.info(
            "application_running",
            message="Press Ctrl+C to stop.",
        )

        await shutdown_event.wait()

        logger.info("shutdown_requested")

    finally:
        shutdown_event.set()

        if health_task is not None:
            await stop_background_task(health_task)

        await market_data_service.stop()
        await stop_background_task(service_task)

        if storage_worker is not None and storage_task is not None:
            await storage_worker.stop()

            try:
                await asyncio.wait_for(
                    storage_task,
                    timeout=(config.storage.shutdown_timeout_seconds),
                )
            except TimeoutError:
                logger.error(
                    "ticker_storage_shutdown_timeout",
                    queue_size=storage_worker.queue_size,
                )

                storage_task.cancel()

                with suppress(asyncio.CancelledError):
                    await storage_task

    logger.info("application_stopped")
    return 0


def run() -> int:
    try:
        return asyncio.run(run_async())
    except KeyboardInterrupt:
        return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

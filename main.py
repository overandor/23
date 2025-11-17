import asyncio
import logging
import signal
import sys

from client import ClientTicket
from config import DRIFT_MARKETS
from drift import DriftTicket
from markets import MarketTicket
from orders import OrderTicket
from shutdown import ShutdownTicket
from wallet import WalletTicket


class TraderCore:
    def __init__(self) -> None:
        self.logger = self._setup_logger()
        self.wallet = WalletTicket(self.logger)
        self.client = ClientTicket()
        self.drift = DriftTicket(self.logger)
        self.markets = MarketTicket(self.drift, self.logger)
        self.orders = None
        self.shutdown = None
        self.is_running = False

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("micro_trader.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        return logging.getLogger("TraderCore")

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, *_args) -> None:
        self.logger.info("Shutdown signal received")
        self.is_running = False

    async def run(self) -> None:
        self._setup_signals()
        if not self.wallet.load():
            return

        await self.drift.init(self.wallet, self.client)
        self.orders = OrderTicket(self.drift, self.drift.provider, self.logger)
        self.shutdown = ShutdownTicket(self.orders, DRIFT_MARKETS, self.logger)
        self.markets.init_symbols(DRIFT_MARKETS)

        self.is_running = True
        self.logger.info("TraderCore initialized. Entering loop.")
        while self.is_running:
            try:
                await asyncio.sleep(30)
            except Exception as exc:
                self.logger.error("Main loop error: %s", exc)
                await asyncio.sleep(10)

        if self.shutdown:
            await self.shutdown.close_all()
        await self.client.close()
        await self.drift.close()


def main() -> None:
    asyncio.run(TraderCore().run())


if __name__ == "__main__":
    main()

from decimal import Decimal
from typing import Dict


class ShutdownTicket:
    """Emergency cleanup workflow."""

    def __init__(self, order_ticket, market_config: Dict[str, Dict], logger) -> None:
        self.order_ticket = order_ticket
        self.market_config = market_config
        self.logger = logger

    async def close_all(self) -> None:
        self.logger.warning("Emergency shutdown initiated")
        for symbol, cfg in self.market_config.items():
            if cfg.get("enabled"):
                await self.order_ticket.place(symbol, Decimal("0"), "long", self.market_config)
        self.logger.info("Shutdown sequence complete")

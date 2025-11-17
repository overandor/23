import time
from collections import deque
from decimal import Decimal
from typing import Deque, Dict, Optional

from driftpy.accounts import get_perp_market_account


class MarketTicket:
    """Fetches Drift perp market data and maintains rolling history."""

    def __init__(self, drift_ticket, logger) -> None:
        self.drift = drift_ticket
        self.logger = logger
        self.history: Dict[str, Deque[Dict]] = {}

    def init_symbols(self, market_config: Dict[str, Dict]) -> None:
        for symbol, cfg in market_config.items():
            if cfg.get("enabled"):
                self.history[symbol] = deque(maxlen=50)

    async def get_data(self, symbol: str, market_config: Dict[str, Dict]) -> Optional[Dict]:
        try:
            market_index = market_config[symbol]["index"]
            market_acct = await get_perp_market_account(self.drift.client.program, market_index)
            oracle = self.drift.client.get_oracle_price_for_perp_market(market_index)
            mark_price = Decimal(str(float(self.drift.client.calculate_mark_price(market_index)) / 1e6))
            data = {
                "symbol": symbol,
                "market_index": market_index,
                "oracle_price": Decimal(str(float(oracle.price) / 1e6)),
                "mark_price": mark_price,
                "funding_rate": market_acct.amm.last_funding_rate,
                "open_interest": market_acct.amm.open_interest,
                "timestamp": Decimal(str(time.time())),
            }
            self.history.setdefault(symbol, deque(maxlen=50)).append(data)
            return data
        except Exception as exc:
            self.logger.error("%s market data error: %s", symbol, exc)
            return None

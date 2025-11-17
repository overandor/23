from decimal import Decimal
from typing import Dict

from driftpy.types import MarketType, OrderParams, OrderTriggerCondition, OrderType, PositionDirection, PostOnlyOption


class OrderTicket:
    """Places Drift perp orders."""

    def __init__(self, drift_ticket, provider, logger) -> None:
        self.drift = drift_ticket
        self.provider = provider
        self.logger = logger

    async def place(self, symbol: str, size: Decimal, side: str, market_config: Dict[str, Dict]) -> bool:
        try:
            market_index = market_config[symbol]["index"]
            direction = PositionDirection.Long() if side.lower() == "long" else PositionDirection.Short()
            base_amount = int(float(size) * 1e9)
            if base_amount < 1000:
                self.logger.warning("Order size below threshold, skipped")
                return False

            params = OrderParams(
                order_type=OrderType.Market(),
                market_type=MarketType.Perp(),
                direction=direction,
                user_order_id=0,
                base_asset_amount=base_amount,
                price=0,
                market_index=market_index,
                reduce_only=False,
                post_only=PostOnlyOption.None(),
                immediate_or_cancel=False,
                trigger_price=0,
                trigger_condition=OrderTriggerCondition.Above(),
                oracle_price_offset=0,
            )
            tx = await self.drift.client.place_order(params)
            sig = await self.provider.send(tx)
            self.logger.info("Order executed %s %s: %s", side, symbol, sig)
            return True
        except Exception as exc:
            self.logger.error("Order failed for %s: %s", symbol, exc)
            return False

import logging
from typing import Optional

from anchorpy import Provider, Wallet
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from driftpy.constants.config import configs
from driftpy.drift_client import DriftClient


class DriftTicket:
    """Initializes and maintains Drift client state."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.client: Optional[DriftClient] = None
        self.provider: Optional[Provider] = None

    async def init(self, wallet_ticket, client_ticket) -> None:
        wallet = Wallet(wallet_ticket.keypair)
        opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
        self.provider = Provider(client_ticket.client, wallet, opts)
        config = configs["mainnet"]
        self.client = DriftClient(self.provider, config)
        try:
            await self.client.get_user_account()
        except Exception:
            tx = await self.client.initialize_user(sub_account_id=0)
            sig = await self.provider.send(tx)
            self.logger.info("Created Drift account: %s", sig)

    async def close(self) -> None:
        if self.client is None:
            return
        try:
            await self.client.unsubscribe()
        except Exception:
            pass

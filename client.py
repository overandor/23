from solana.rpc.async_api import AsyncClient


class ClientTicket:
    """Async Solana RPC client lifecycle wrapper."""

    def __init__(self, rpc_url: str = "https://api.mainnet-beta.solana.com") -> None:
        self.rpc_url = rpc_url
        self.client = AsyncClient(self.rpc_url)

    async def close(self) -> None:
        await self.client.close()

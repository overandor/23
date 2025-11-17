import json
import logging
import os
from typing import Optional

from solders.keypair import Keypair


class WalletTicket:
    """Loads a Solana keypair from SOLANA_PRIVATE_KEY."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.keypair: Optional[Keypair] = None

    def load(self) -> bool:
        """Load a keypair from env. Supports base58 or JSON secret format."""
        try:
            pk_str = os.getenv("SOLANA_PRIVATE_KEY")
            if not pk_str:
                self.logger.error("Missing SOLANA_PRIVATE_KEY env var")
                return False

            if pk_str.startswith("["):
                secret = bytes(json.loads(pk_str))
            else:
                secret = Keypair.from_base58_string(pk_str).secret()

            self.keypair = Keypair.from_bytes(bytes(secret))
            self.logger.info("Wallet loaded: %s", self.keypair.pubkey())
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Wallet load error: %s", exc)
            return False

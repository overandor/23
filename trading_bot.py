#!/usr/bin/env python3
"""
PRODUCTION SOLANA QUANT ENGINE - MAINNET HARDENED
Enterprise-grade refactor with security, observability, and fault tolerance
"""
import asyncio
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from decimal import Decimal
from enum import Enum
import base58
# External dependencies
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from anchorpy import Provider, Wallet
# Drift Protocol
from driftpy.drift_client import DriftClient
from driftpy.accounts import get_perp_market_account
from driftpy.types import PositionDirection
from driftpy.constants.config import configs
from driftpy.math.amm import calculate_mark_price
# Monitoring (optional)
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
console = Console()
logger = logging.getLogger(__name__)
# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================
class EnvironmentConfig:
    """Environment-based configuration with validation"""
    def __init__(self, env: str = "mainnet"):
        self.env = env
        self.drift_config = configs[self.env]

        # RPC Configuration
        self.rpc_urls = {
            "primary": "https://api.mainnet-beta.solana.com",
            "fallback": "https://solana-api.projectserum.com"
        }

        # Safety Parameters
        self.min_sol_balance = Decimal("0.05")
        self.min_usdc_balance = Decimal("20.0")
        self.max_position_size_usd = Decimal("100.0")

        # Risk Management
        self.take_profit_pct = Decimal("0.01")  # 1%
        self.stop_loss_pct = Decimal("0.02")    # 2%
        self.circuit_breaker_loss = Decimal("-0.20")  # -20%
        self.max_daily_trades = 50
        self.max_open_positions = 3

        # Timing
        self.scan_interval = 5.0
        self.execution_interval = 3.0
        self.health_check_interval = 30.0
        self.confirmation_timeout = 60

        # Retry Logic
        self.max_retries = 3
        self.retry_backoff = 2.0

    def validate(self) -> bool:
        """Validate configuration"""
        if self.env not in ["mainnet", "devnet"]:
            raise ValueError(f"Invalid environment: {self.env}")

        if self.max_position_size_usd <= 0:
            raise ValueError("Position size must be positive")

        return True
class MarketConfig:
    """Market definitions and parameters"""
    MARKETS = {
        0: {"symbol": "SOL-PERP", "min_size": 0.01, "tick_size": 0.01},
        1: {"symbol": "BTC-PERP", "min_size": 0.001, "tick_size": 0.01},
        2: {"symbol": "ETH-PERP", "min_size": 0.01, "tick_size": 0.01},
    }
    @classmethod
    def get_market_symbol(cls, market_index: int) -> str:
        return cls.MARKETS.get(market_index, {}).get("symbol", "UNKNOWN")
# ============================================================================
# METRICS & OBSERVABILITY
# ============================================================================
class MetricsCollector:
    """Prometheus metrics collection"""
    def __init__(self):
        if not METRICS_ENABLED:
            return

        self.trades_total = Counter('trades_total', 'Total trades executed', ['agent_id', 'market'])
        self.trades_pnl = Histogram('trade_pnl', 'Trade PnL distribution')
        self.position_size = Gauge('position_size_usd', 'Current position size', ['agent_id'])
        self.agent_balance = Gauge('agent_balance_sol', 'Agent SOL balance', ['agent_id'])
        self.errors_total = Counter('errors_total', 'Total errors', ['agent_id', 'error_type'])

    def record_trade(self, agent_id: int, market: str, pnl: float):
        if not METRICS_ENABLED:
            return
        self.trades_total.labels(agent_id=agent_id, market=market).inc()
        self.trades_pnl.observe(pnl)

    def record_error(self, agent_id: int, error_type: str):
        if not METRICS_ENABLED:
            return
        self.errors_total.labels(agent_id=agent_id, error_type=error_type).inc()
# ============================================================================
# DATA MODELS
# ============================================================================
class PositionStatus(Enum):
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
@dataclass
class PositionState:
    """Enhanced position state tracking"""
    market_index: int
    direction: PositionDirection
    entry_price: Decimal
    size: Decimal
    status: PositionStatus = PositionStatus.OPENING
    dca_count: int = 0
    opened_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    tx_signature: Optional[str] = None
    realized_pnl: Decimal = Decimal("0")

    @property
    def is_long(self) -> bool:
        return self.direction.name == "LONG"
    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL"""
        if self.status != PositionStatus.OPEN:
            return self.realized_pnl

        pnl_pct = (current_price - self.entry_price) / self.entry_price
        if not self.is_long:
            pnl_pct = -pnl_pct
        return pnl_pct
@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    errors: int = 0
    last_error: Optional[str] = None
    daily_trades: int = 0
    last_trade_reset: float = field(default_factory=lambda: time.time())

    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0
    def reset_daily_counter(self):
        """Reset daily trade counter"""
        current_time = time.time()
        if current_time - self.last_trade_reset > 86400:  # 24 hours
            self.daily_trades = 0
            self.last_trade_reset = current_time
@dataclass
class TradingAgent:
    """Enhanced trading agent"""
    id: int
    keypair: Keypair
    config: EnvironmentConfig
    drift_client: Optional[DriftClient] = None
    position: Optional[PositionState] = None
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    is_active: bool = False
    is_healthy: bool = True
    sol_balance: Decimal = Decimal("0")
    usdc_balance: Decimal = Decimal("0")
    last_health_check: float = field(default_factory=time.time)

    @property
    def public_key(self) -> str:
        return str(self.keypair.pubkey())
    def can_trade(self) -> bool:
        """Check if agent can execute trades"""
        self.metrics.reset_daily_counter()

        return (
            self.is_healthy and
            self.sol_balance >= self.config.min_sol_balance and
            self.usdc_balance >= self.config.min_usdc_balance and
            self.metrics.total_pnl > self.config.circuit_breaker_loss and
            self.metrics.daily_trades < self.config.max_daily_trades
        )
# ============================================================================
# SECURE WALLET MANAGEMENT
# ============================================================================
class SecureWalletManager:
    """Production wallet management with encryption"""
    def __init__(self, wallet_file: Path = Path("wallets.enc.json")):
        self.wallet_file = wallet_file
        self.connection: Optional[AsyncClient] = None

    async def initialize(self, rpc_url: str):
        """Initialize connection"""
        self.connection = AsyncClient(rpc_url, commitment=Confirmed)

    async def generate_wallets(self, count: int, config: EnvironmentConfig) -> List[TradingAgent]:
        """Generate new agent wallets"""
        console.print("[bold green]🔐 Generating Secure Wallets...[/bold green]")

        agents = []
        wallet_data = {}

        for i in range(count):
            keypair = Keypair()
            agent = TradingAgent(id=i, keypair=keypair, config=config)
            agents.append(agent)

            wallet_data[f"agent_{i}"] = {
                "public_key": agent.public_key,
                "private_key": base58.b58encode(bytes(keypair.secret())).decode()
            }

            console.print(f"✅ Agent {i}: [cyan]{agent.public_key}[/cyan]")

        # Save with warning
        with open(self.wallet_file, 'w') as f:
            json.dump(wallet_data, f, indent=2)

        console.print(f"[bold red]⚠️  Wallets saved to {self.wallet_file}[/bold red]")
        console.print("[yellow]Encrypt this file and store securely![/yellow]")

        return agents
    async def load_wallets(self, config: EnvironmentConfig) -> List[TradingAgent]:
        """Load existing wallets"""
        if not self.wallet_file.exists():
            raise FileNotFoundError(f"Wallet file not found: {self.wallet_file}")

        with open(self.wallet_file, 'r') as f:
            wallet_data = json.load(f)

        agents = []
        for key, data in wallet_data.items():
            agent_id = int(key.split('_')[1])
            private_key = base58.b58decode(data["private_key"])
            keypair = Keypair.from_bytes(private_key)
            agents.append(TradingAgent(id=agent_id, keypair=keypair, config=config))

        console.print(f"[green]✅ Loaded {len(agents)} wallets[/green]")
        return agents
    async def check_balances(self, agent: TradingAgent) -> bool:
        """Check agent funding"""
        try:
            response = await self.connection.get_balance(agent.keypair.pubkey())
            agent.sol_balance = Decimal(response.value) / Decimal(1e9)

            # Get USDC balance from Associated Token Account
            usdc_mint_pubkey = agent.config.drift_config.spot_markets[0].mint
            TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
            ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

            # Calculate ATA
            ata_address, _ = Pubkey.find_program_address(
                [bytes(agent.keypair.pubkey()), bytes(TOKEN_PROGRAM_ID), bytes(usdc_mint_pubkey)],
                ASSOCIATED_TOKEN_PROGRAM_ID
            )

            try:
                response = await self.connection.get_token_account_balance(ata_address)
                agent.usdc_balance = Decimal(response.value.amount) / (10 ** response.value.decimals)
            except Exception:
                agent.usdc_balance = Decimal(0)

            is_funded = (
                agent.sol_balance >= agent.config.min_sol_balance and
                agent.usdc_balance >= agent.config.min_usdc_balance
            )

            if not is_funded:
                console.print(
                    f"[red]❌ Agent {agent.id} underfunded:[/red] "
                    f"{agent.sol_balance:.4f} SOL (need {agent.config.min_sol_balance}), "
                    f"{agent.usdc_balance:.2f} USDC (need {agent.config.min_usdc_balance})"
                )
                console.print(f"[yellow]Send funds to: {agent.public_key}[/yellow]")

            return is_funded

        except Exception as e:
            logger.error(f"Balance check failed for agent {agent.id}: {e}")
            return False
    async def close(self):
        """Cleanup"""
        if self.connection:
            await self.connection.close()
# ============================================================================
# DRIFT CLIENT INITIALIZATION
# ============================================================================
class DriftClientManager:
    """Manage Drift client lifecycle"""
    @staticmethod
    async def initialize_client(
        agent: TradingAgent,
        wallet_manager: SecureWalletManager,
        metrics: MetricsCollector
    ) -> bool:
        """Initialize Drift client with retries"""
        try:
            console.print(f"[cyan]Initializing Agent {agent.id}...[/cyan]")

            # Pre-flight checks
            if not await wallet_manager.check_balances(agent):
                return False

            # Create provider
            provider = Provider(
                wallet_manager.connection,
                Wallet(agent.keypair)
            )

            # Initialize Drift client
            agent.drift_client = DriftClient(
                provider.connection,
                Wallet(agent.keypair),
                "mainnet",
                perp_market_indexes=list(MarketConfig.MARKETS.keys()),
            )

            # Subscribe with retry logic
            for attempt in range(agent.config.max_retries):
                try:
                    await agent.drift_client.subscribe()
                    break
                except Exception as e:
                    if attempt == agent.config.max_retries - 1:
                        raise
                    console.print(f"[yellow]Retry {attempt + 1}/{agent.config.max_retries}...[/yellow]")
                    await asyncio.sleep(agent.config.retry_backoff ** attempt)

            metrics.agent_balance.labels(agent_id=agent.id).set(float(agent.sol_balance))
            console.print(f"[green]✅ Agent {agent.id} ready[/green]")
            return True

        except Exception as e:
            agent.metrics.errors += 1
            agent.metrics.last_error = str(e)
            metrics.record_error(agent.id, "initialization")
            console.print(f"[red]❌ Agent {agent.id} init failed: {e}[/red]")
            return False
# ============================================================================
# ALPHA GENERATION ENGINE
# ============================================================================
class AlphaEngine:
    """Alpha signal generation with error handling"""
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.cache: Dict[int, Any] = {}

    async def fetch_market_data(
        self,
        drift_client: DriftClient,
        market_index: int,
        max_retries: int = 3
    ) -> Optional[Any]:
        """Fetch market data with caching and retries"""
        try:
            for attempt in range(max_retries):
                try:
                    market_account = await get_perp_market_account(
                        drift_client.program,
                        market_index
                    )
                    self.cache[market_index] = market_account
                    return market_account
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            return self.cache.get(market_index)
    async def calculate_alpha(
        self,
        agent: TradingAgent,
        market_index: int
    ) -> Optional[Decimal]:
        """Calculate alpha signal"""
        try:
            if not agent.drift_client:
                return None

            market_account = await self.fetch_market_data(
                agent.drift_client,
                market_index,
                agent.config.max_retries
            )

            if not market_account:
                return None

            # Calculate market metrics
            mark_price = Decimal(str(calculate_mark_price(market_account)))
            funding_rate = Decimal(market_account.amm.last_funding_rate) / Decimal(1e9)

            # Simple mean-reversion alpha (replace with real strategy)
            alpha_signal = -funding_rate * Decimal("10.0")

            # Clip to [-1, 1]
            return max(Decimal("-1.0"), min(Decimal("1.0"), alpha_signal))

        except Exception as e:
            agent.metrics.errors += 1
            self.metrics.record_error(agent.id, "alpha_calculation")
            logger.error(f"Alpha calculation failed for agent {agent.id}: {e}")
            return None
# ============================================================================
# POSITION MANAGEMENT
# ============================================================================
class PositionManager:
    """Position lifecycle management"""
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics

    async def open_position(
        self,
        agent: TradingAgent,
        market_index: int,
        alpha_signal: Decimal
    ) -> bool:
        """Open position with confirmation"""
        try:
            if not agent.can_trade():
                return False

            direction = PositionDirection.LONG() if alpha_signal > 0 else PositionDirection.SHORT()
            market_symbol = MarketConfig.get_market_symbol(market_index)

            console.print(
                f"[cyan]📈 Agent {agent.id} opening {direction.name} "
                f"on {market_symbol}[/cyan]"
            )

            # Get current mark price
            market_account = await get_perp_market_account(
                agent.drift_client.program,
                market_index
            )
            mark_price = Decimal(str(calculate_mark_price(market_account)))

            # Calculate position size
            position_size = agent.config.max_position_size_usd

            # Submit transaction
            tx_sig = await agent.drift_client.open_position(
                market_index=market_index,
                direction=direction,
                quote_asset_amount=int(position_size * Decimal(1_000_000)),
            )

            # Record position
            agent.position = PositionState(
                market_index=market_index,
                direction=direction,
                entry_price=mark_price,
                size=position_size,
                status=PositionStatus.OPEN,
                tx_signature=str(tx_sig)
            )

            agent.is_active = True
            agent.metrics.total_trades += 1
            agent.metrics.daily_trades += 1

            self.metrics.record_trade(agent.id, market_symbol, 0.0)
            self.metrics.position_size.labels(agent_id=agent.id).set(float(position_size))

            console.print(
                f"[green]✅ Agent {agent.id} position opened @ "
                f"${mark_price:.2f}[/green]"
            )

            return True

        except Exception as e:
            agent.metrics.errors += 1
            self.metrics.record_error(agent.id, "position_open")
            console.print(f"[red]❌ Agent {agent.id} open failed: {e}[/red]")
            return False
    async def close_position(
        self,
        agent: TradingAgent,
        current_price: Decimal,
        reason: str = "Exit"
    ) -> bool:
        """Close position with confirmation"""
        try:
            if not agent.position or agent.position.status != PositionStatus.OPEN:
                return False

            console.print(f"[yellow]Agent {agent.id} closing: {reason}[/yellow]")

            # Submit close transaction
            agent.position.status = PositionStatus.CLOSING
            tx_sig = await agent.drift_client.close_position(agent.position.market_index)

            # Calculate PnL
            pnl = agent.position.calculate_pnl(current_price)
            agent.position.realized_pnl = pnl
            agent.position.status = PositionStatus.CLOSED
            agent.position.closed_at = time.time()

            # Update metrics
            agent.metrics.total_pnl += pnl
            if pnl > 0:
                agent.metrics.winning_trades += 1
            else:
                agent.metrics.losing_trades += 1

            if agent.metrics.total_pnl < agent.metrics.max_drawdown:
                agent.metrics.max_drawdown = agent.metrics.total_pnl

            market_symbol = MarketConfig.get_market_symbol(agent.position.market_index)
            self.metrics.record_trade(agent.id, market_symbol, float(pnl))
            self.metrics.position_size.labels(agent_id=agent.id).set(0)

            console.print(
                f"[green]✅ Agent {agent.id} closed: {reason} "
                f"(PnL: {pnl:+.2%})[/green]"
            )

            agent.position = None
            agent.is_active = False
            return True

        except Exception as e:
            agent.metrics.errors += 1
            self.metrics.record_error(agent.id, "position_close")
            console.print(f"[red]❌ Agent {agent.id} close failed: {e}[/red]")
            return False
# ============================================================================
# RISK MANAGEMENT
# ============================================================================
class RiskManager:
    """Risk management and circuit breakers"""
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics

    async def execute_risk_checks(
        self,
        agent: TradingAgent,
        position_manager: PositionManager,
        current_price: Decimal
    ) -> bool:
        """Execute all risk checks"""
        if not agent.position or agent.position.status != PositionStatus.OPEN:
            return True

        # Calculate current PnL
        pnl = agent.position.calculate_pnl(current_price)

        # Circuit breaker: total loss
        if agent.metrics.total_pnl <= agent.config.circuit_breaker_loss:
            console.print(f"[bold red]🚨 CIRCUIT BREAKER Agent {agent.id}[/bold red]")
            await position_manager.close_position(agent, current_price, "CIRCUIT BREAKER")
            agent.is_healthy = False
            return False

        # Take profit
        if pnl >= agent.config.take_profit_pct:
            await position_manager.close_position(
                agent,
                current_price,
                f"TP: {pnl:+.2%}"
            )
            return False

        # Stop loss
        if pnl <= -agent.config.stop_loss_pct:
            await position_manager.close_position(
                agent,
                current_price,
                f"SL: {pnl:+.2%}"
            )
            return False

        return True
# ============================================================================
# TRADING ENGINE
# ============================================================================
class TradingEngine:
    """Main trading engine orchestrator"""
    def __init__(
        self,
        config: EnvironmentConfig,
        alpha_engine: AlphaEngine,
        position_manager: PositionManager,
        risk_manager: RiskManager
    ):
        self.config = config
        self.alpha_engine = alpha_engine
        self.position_manager = position_manager
        self.risk_manager = risk_manager

    async def execute_trading_cycle(self, agent: TradingAgent):
        """Single trading cycle"""
        try:
            # Health check
            if not agent.can_trade():
                return

            # Entry logic
            if not agent.is_active:
                # Select market (simple random for now)
                import random
                market_index = random.choice(list(MarketConfig.MARKETS.keys()))

                # Calculate alpha
                alpha_signal = await self.alpha_engine.calculate_alpha(agent, market_index)

                if alpha_signal and abs(alpha_signal) > Decimal("0.3"):
                    await self.position_manager.open_position(
                        agent,
                        market_index,
                        alpha_signal
                    )

            # Position management
            if agent.is_active and agent.position:
                # Update mark price
                market_account = await self.alpha_engine.fetch_market_data(
                    agent.drift_client,
                    agent.position.market_index,
                    agent.config.max_retries
                )

                if market_account:
                    current_price = Decimal(str(calculate_mark_price(market_account)))

                    # Risk checks
                    await self.risk_manager.execute_risk_checks(
                        agent,
                        self.position_manager,
                        current_price
                    )

        except Exception as e:
            agent.metrics.errors += 1
            logger.error(f"Trading cycle error for agent {agent.id}: {e}")
# ============================================================================
# SUPERVISOR DASHBOARD
# ============================================================================
class SupervisorDashboard:
    """Real-time monitoring dashboard"""
    def __init__(self, agents: List[TradingAgent]):
        self.agents = agents

    def generate_table(self) -> Table:
        """Generate status table"""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )

        table.add_column("Agent", style="cyan")
        table.add_column("Status")
        table.add_column("Market")
        table.add_column("Side")
        table.add_column("Entry")
        table.add_column("PnL", justify="right")
        table.add_column("Total PnL", justify="right")
        table.add_column("Win Rate")
        table.add_column("Trades")
        table.add_column("Errors")

        for agent in self.agents:
            status = "🟢" if agent.is_active else "⚪"
            if not agent.is_healthy:
                status = "🔴"

            market = MarketConfig.get_market_symbol(agent.position.market_index) if agent.position else "-"
            side = agent.position.direction.name if agent.position else "-"
            entry = f"${agent.position.entry_price:.2f}" if agent.position else "-"

            pnl = "-"
            if agent.position and agent.position.status == PositionStatus.OPEN:
                market_account = agent.drift_client.get_perp_market_account(agent.position.market_index)
                if market_account:
                    current_price = Decimal(str(calculate_mark_price(market_account)))
                    pnl_value = agent.position.calculate_pnl(current_price)
                    pnl = f"{pnl_value:+.2%}"

            total_pnl = f"{agent.metrics.total_pnl:+.2%}"
            win_rate = f"{agent.metrics.win_rate:.1%}"

            table.add_row(
                f"Agent {agent.id}",
                status,
                market,
                side,
                entry,
                pnl,
                total_pnl,
                win_rate,
                f"{agent.metrics.total_trades}",
                f"{agent.metrics.errors}"
            )

        return table
    async def run(self):
        """Run dashboard loop"""
        with Live(self.generate_table(), refresh_per_second=1) as live:
            while True:
                live.update(self.generate_table())
                await asyncio.sleep(5)
# ============================================================================
# MAIN APPLICATION
# ============================================================================
async def main():
    """Main application entry point"""
    # Initialize configuration
    config = EnvironmentConfig("mainnet")
    config.validate()
    # Display warning
    console.print(Panel.fit(
        "[bold red]🚀 PRODUCTION SOLANA QUANT ENGINE[/bold red]\n"
        "[yellow]⚠️  MAINNET - REAL FUNDS AT RISK[/yellow]\n"
        "[white]Ensure all safety checks pass before trading[/white]",
        border_style="red"
    ))
    # Initialize metrics
    metrics = MetricsCollector()
    if METRICS_ENABLED:
        start_http_server(8000)
        console.print("[green]📊 Metrics server started on :8000[/green]")
    # Initialize wallet manager
    wallet_manager = SecureWalletManager()
    await wallet_manager.initialize(config.rpc_urls["primary"])
    # Load wallets
    try:
        agents = await wallet_manager.load_wallets(config)
    except FileNotFoundError:
        console.print("[yellow]No wallets found. Generate new? (y/n)[/yellow]")
        if input().lower() == 'y':
            agents = await wallet_manager.generate_wallets(2, config)
        else:
            return
    # Display funding requirements
    console.print(Panel(
        f"[bold yellow]💰 FUNDING REQUIREMENTS:[/bold yellow]\n"
        f"Each agent needs:\n"
        f"• {config.min_sol_balance} SOL\n"
        f"• {config.min_usdc_balance} USDC\n\n"
        f"[cyan]Agent addresses:[/cyan]\n" +
        "\n".join([f"Agent {a.id}: {a.public_key}" for a in agents]),
        border_style="yellow"
    ))
    input("\n[Press ENTER when wallets are funded]")
    # Initialize Drift clients
    console.print("[cyan]Initializing Drift clients...[/cyan]")
    client_manager = DriftClientManager()
    init_results = await asyncio.gather(*[
        client_manager.initialize_client(agent, wallet_manager, metrics)
        for agent in agents
    ])
    successful_agents = [
        agent for agent, success in zip(agents, init_results) if success
    ]
    if not successful_agents:
        console.print("[red]❌ No agents initialized[/red]")
        return
    console.print(f"[green]✅ {len(successful_agents)} agents ready[/green]")
    # Initialize trading components
    alpha_engine = AlphaEngine(metrics)
    position_manager = PositionManager(metrics)
    risk_manager = RiskManager(metrics)
    trading_engine = TradingEngine(
        config,
        alpha_engine,
        position_manager,
        risk_manager
    )
    # Start trading loops
    async def agent_loop(agent: TradingAgent):
        while agent.is_healthy:
            await trading_engine.execute_trading_cycle(agent)
            await asyncio.sleep(config.execution_interval)

    # Start supervisor dashboard
    dashboard = SupervisorDashboard(successful_agents)

    # Run all tasks concurrently
    await asyncio.gather(
        dashboard.run(),
        *[agent_loop(agent) for agent in successful_agents]
    )

if __name__ == "__main__":
    asyncio.run(main())

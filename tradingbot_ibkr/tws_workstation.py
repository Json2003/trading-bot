"""Account-checked, read-only access to local Interactive Brokers TWS."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

TWS_HOST = "127.0.0.1"
WATCHLIST = ("IBM", "JPM", "KO")
_ACCOUNT = re.compile(r"[A-Z][A-Z0-9]{4,19}")
_DEFAULTS = {"paper": (7497, 71), "live": (7496, 72)}
_TAGS = {"NetLiquidation": "equity", "SettledCash": "settled_cash", "AvailableFunds": "available_funds"}


class AccountMismatch(RuntimeError):
    """The configured account was not advertised by the connected TWS session."""


@dataclass(frozen=True)
class Profile:
    mode: str
    port: int
    client_id: int
    account: str

    @property
    def masked_account(self) -> str | None:
        return f"••••{self.account[-4:]}" if self.account else None


def _number(value: object) -> float | None:
    try:
        result = float(str(value))
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and abs(result) < 1e50 else None


class TwsAccountReader:
    """Verify an explicit account and read one snapshot on request.

    Paper/live ports are configurable, never proof of identity. The caller
    configures the expected account for each session. This class never accepts
    an order, registers a trading engine, or keeps a broker session active.
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._env = dict(os.environ if environ is None else environ)
        self._lock = asyncio.Lock()
        self._last_check: dict[str, float] = {}

    def _profile(self, mode: str) -> Profile:
        port, client_id = _DEFAULTS[mode]
        prefix = f"IBKR_TWS_{mode.upper()}"
        try:
            port = int(self._env.get(f"{prefix}_PORT", str(port)))
            client_id = int(self._env.get(f"{prefix}_CLIENT_ID", str(client_id)))
        except ValueError:
            raise ValueError(f"{prefix} port and client ID must be integers") from None
        if not 1 <= port <= 65535 or not 1 <= client_id <= 2_147_483_647:
            raise ValueError(f"{prefix} needs a valid port and a nonzero client ID")
        account = self._env.get(f"{prefix}_ACCOUNT", "").strip().upper()
        if account and not _ACCOUNT.fullmatch(account):
            raise ValueError(f"{prefix}_ACCOUNT has an invalid format")
        return Profile(mode, port, client_id, account)

    def _profiles(self) -> dict[str, Profile]:
        profiles = {mode: self._profile(mode) for mode in _DEFAULTS}
        paper, live = profiles["paper"], profiles["live"]
        if paper.account and live.account and (paper.port == live.port or paper.account == live.account):
            raise ValueError("Paper and Live must use separate TWS ports and account IDs")
        return profiles

    @staticmethod
    def _sdk_version() -> str | None:
        try:
            return version("ib_async")
        except PackageNotFoundError:
            return None

    def settings(self) -> dict[str, object]:
        sdk = self._sdk_version()
        try:
            profiles = self._profiles()
        except ValueError as exc:
            return {"profiles": {}, "sdk_version": sdk, "error": str(exc)}
        return {
            "profiles": {
                mode: {
                    "port": profile.port,
                    "client_id": profile.client_id,
                    "account": profile.masked_account,
                    "configured": bool(profile.account),
                }
                for mode, profile in profiles.items()
            },
            "sdk_version": sdk,
            "watchlist": WATCHLIST,
        }

    async def check(self, mode: str, symbol: str) -> dict[str, object]:
        if mode not in _DEFAULTS or symbol not in WATCHLIST:
            raise ValueError("Select a TWS workspace and a supported NYSE symbol")
        profile = self._profiles()[mode]
        if not profile.account:
            raise RuntimeError(f"Set IBKR_TWS_{mode.upper()}_ACCOUNT and restart the local operator")
        sdk = self._sdk_version()
        if sdk is None:
            raise RuntimeError('TWS client is missing. Install the project with the "tws" extra.')
        if self._lock.locked():
            raise RuntimeError("A TWS check is already running")
        async with self._lock:
            now = time.monotonic()
            if now - self._last_check.get(mode, -math.inf) < 5:
                raise RuntimeError("Wait five seconds before checking this TWS session again")
            self._last_check[mode] = now
            try:
                return await asyncio.wait_for(self._read(profile, symbol, sdk), timeout=15)
            except (asyncio.TimeoutError, TimeoutError):
                raise RuntimeError("TWS did not finish the account check in time") from None
            except (ConnectionError, OSError):
                raise RuntimeError(
                    f"TWS is unreachable on 127.0.0.1:{profile.port}. Check its login and socket settings."
                ) from None
            except AccountMismatch:
                raise
            except Exception:
                raise RuntimeError("TWS could not complete the read-only account check") from None

    async def _read(self, profile: Profile, symbol: str, sdk: str) -> dict[str, object]:
        from ib_async import IB, StartupFetchNONE, Stock

        client = IB()
        try:
            await client.connectAsync(
                TWS_HOST, profile.port, clientId=profile.client_id, timeout=4, readonly=True,
                raiseSyncErrors=True, fetchFields=StartupFetchNONE,
            )
            if profile.account not in client.managedAccounts():
                raise AccountMismatch("The TWS session did not report the expected account. Check the login and port.")
            accounts = await asyncio.wait_for(client.accountSummaryAsync(profile.account), timeout=4)
            balances = {}
            for tag, label in _TAGS.items():
                rows = [(row.currency, _number(row.value)) for row in accounts if row.account == profile.account and row.tag == tag]
                # Do not label account-base or multi-currency values as USD.
                if len(rows) == 1 and rows[0][1] is not None:
                    balances[label] = {"currency": rows[0][0], "value": rows[0][1]}
            positions = []
            for row in client.positions(profile.account):
                if row.account != profile.account:
                    continue
                quantity = _number(row.position)
                if quantity:
                    positions.append({
                        "symbol": row.contract.localSymbol or row.contract.symbol,
                        "security_type": row.contract.secType,
                        "currency": row.contract.currency,
                        "quantity": quantity,
                        "average_price": _number(row.avgCost),
                    })
            captured = datetime.now(timezone.utc).isoformat()
            bars: list[dict[str, object]] = []
            chart_message = None
            try:
                contract = Stock(symbol, "SMART", "USD", primaryExchange="NYSE")
                quotes = await client.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr="2 D", barSizeSetting="5 mins",
                    whatToShow="TRADES", useRTH=True, formatDate=2, timeout=5,
                )
                for bar in quotes:
                    if not isinstance(bar.date, datetime) or bar.date.utcoffset() is None:
                        continue
                    o, h, low, c = (_number(getattr(bar, key)) for key in ("open", "high", "low", "close"))
                    if o is None or h is None or low is None or c is None:
                        continue
                    if not (0 < low <= min(o, c) <= max(o, c) <= h):
                        continue
                    volume = _number(bar.volume)
                    bars.append({
                        "time": bar.date.astimezone(timezone.utc).isoformat(),
                        "open": o, "high": h, "low": low, "close": c,
                        "volume": volume if volume is not None and volume >= 0 else None,
                    })
                if not bars:
                    chart_message = "No historical bars returned. Check TWS market data permissions."
            except Exception:
                chart_message = "TWS historical bars unavailable for this symbol."
            return {
                "mode": profile.mode,
                "account": profile.masked_account,
                "port": profile.port,
                "sdk_version": sdk,
                "server_version": client.client.serverVersion(),
                "captured_at": captured,
                "balances": balances,
                "positions": positions,
                "chart": {"symbol": symbol, "bars": bars, "message": chart_message},
                "execution_enabled": False,
            }
        finally:
            client.disconnect()

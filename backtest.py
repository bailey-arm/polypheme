"""
backtest.py – Polymarket signal backtester

Each market is a binary that settles at 0 (No) or 1 (Yes).
A position of +1 means "long Yes" (profit if price rises toward 1).
A position of -1 means "short Yes" (profit if price falls toward 0).
PnL is mark-to-market using hourly prices, with final settlement
at the last observed price (which is ~0 or ~1 for resolved markets).

Usage:
    python backtest.py                          # compare all built-in signals
    python backtest.py --signal momentum
    python backtest.py --notional 1000 --plot

Adding a custom signal (the only interface you need):

    class MySignal(Signal):
        name = "my_signal"

        def generate(self, history: pd.DataFrame) -> pd.Series:
            # history: DataFrame with columns [timestamp, price]
            #          sorted ascending, covers one token's full history
            # return:  Series indexed by timestamp, values in [-1.0, 1.0]
            #          +1 = fully long, -1 = fully short, 0 = flat
            prices = history.set_index("timestamp")["price"]
            signal = pd.Series(0.0, index=prices.index)
            signal[prices < 0.25] = 1.0
            signal[prices > 0.75] = -1.0
            return signal
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Data loading ──────────────────────────────────────────────────────────────

MARKETS_PATH = "data/markets.parquet"
PRICES_PATH = "data/prices.parquet"


def load_data(yes_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and join price + market data. Returns (prices_df, markets_df)."""
    prices = pd.read_parquet(PRICES_PATH)
    markets = pd.read_parquet(MARKETS_PATH)

    if yes_only:
        markets = markets[markets["token_index"] == 0]

    # Keep only tokens that have price data
    valid = set(prices["token_id"].unique())
    markets = markets[markets["token_id"].isin(valid)].copy()

    return prices, markets


# ── Signal base class ─────────────────────────────────────────────────────────

class Signal:
    """
    Base class for trading signals.

    Subclass this and implement generate(). That's all you need.
    """
    name: str = "base"

    def generate(self, history: pd.DataFrame) -> pd.Series:
        """
        Args:
            history: DataFrame(timestamp, price) for ONE token,
                     sorted ascending, all data up to (and including)
                     the current bar.
        Returns:
            pd.Series indexed by timestamp, values in [-1.0, 1.0]
            +1 = fully long, -1 = fully short, 0 = flat
        """
        raise NotImplementedError


# ── Built-in signals ──────────────────────────────────────────────────────────

class AlwaysLong(Signal):
    """Buy at the first bar and hold. Useful as a baseline."""
    name = "always_long"

    def generate(self, history: pd.DataFrame) -> pd.Series:
        prices = history.set_index("timestamp")["price"]
        return pd.Series(1.0, index=prices.index)


class AlwaysShort(Signal):
    """Sell at the first bar and hold."""
    name = "always_short"

    def generate(self, history: pd.DataFrame) -> pd.Series:
        prices = history.set_index("timestamp")["price"]
        return pd.Series(-1.0, index=prices.index)


class MeanReversion(Signal):
    """
    Buy when price is below its rolling mean minus `threshold` std devs.
    Sell when price is above its rolling mean plus `threshold` std devs.
    Flat otherwise.
    """
    name = "mean_reversion"

    def __init__(self, window: int = 24, threshold: float = 1.0):
        self.window = window
        self.threshold = threshold

    def generate(self, history: pd.DataFrame) -> pd.Series:
        prices = history.set_index("timestamp")["price"]
        roll_mean = prices.rolling(self.window, min_periods=1).mean()
        roll_std = prices.rolling(self.window, min_periods=1).std().fillna(0)

        signal = pd.Series(0.0, index=prices.index)
        signal[prices < roll_mean - self.threshold * roll_std] = 1.0
        signal[prices > roll_mean + self.threshold * roll_std] = -1.0
        return signal


class Momentum(Signal):
    """
    Go long when recent return is positive, short when negative.
    Uses a rolling return over `window` bars.
    """
    name = "momentum"

    def __init__(self, window: int = 12):
        self.window = window

    def generate(self, history: pd.DataFrame) -> pd.Series:
        prices = history.set_index("timestamp")["price"]
        ret = prices.diff(self.window)
        signal = pd.Series(0.0, index=prices.index)
        signal[ret > 0] = 1.0
        signal[ret < 0] = -1.0
        return signal


class ThresholdEntry(Signal):
    """
    Buy (long) when price drops below `buy_below`.
    Sell (short) when price rises above `sell_above`.
    Hold existing position in between.
    """
    name = "threshold"

    def __init__(self, buy_below: float = 0.01, sell_above: float = 0.99):
        self.buy_below = buy_below
        self.sell_above = sell_above

    def generate(self, history: pd.DataFrame) -> pd.Series:
        prices = history.set_index("timestamp")["price"]
        signal = pd.Series(0.0, index=prices.index)
        signal[prices < self.buy_below] = 1.0
        signal[prices > self.sell_above] = -1.0
        # Forward-fill: hold position until a new signal fires
        signal = signal.replace(0.0, np.nan).ffill().fillna(0.0)
        return signal


class FadeTowardsFair(Signal):
    """
    Fade extreme prices: long when below `low`, short when above `high`,
    with position sized proportionally to distance from fair value (0.5).
    """
    name = "fade"

    def __init__(self, low: float = 0.15, high: float = 0.85):
        self.low = low
        self.high = high

    def generate(self, history: pd.DataFrame) -> pd.Series:
        prices = history.set_index("timestamp")["price"]
        signal = pd.Series(0.0, index=prices.index)
        # Long: price below `low`, scaled by distance from 0
        mask_long = prices < self.low
        signal[mask_long] = (self.low - prices[mask_long]) / self.low
        # Short: price above `high`, scaled by distance from 1
        mask_short = prices > self.high
        signal[mask_short] = -(prices[mask_short] - self.high) / (1 - self.high)
        return signal.clip(-1, 1)


ALL_SIGNALS: Dict[str, Signal] = {
    s.name: s
    for s in [
        AlwaysLong(),
        AlwaysShort(),
        MeanReversion(),
        Momentum(),
        ThresholdEntry(),
        FadeTowardsFair(),
    ]
}


# ── Backtester ────────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    token_id: str
    slug: str
    question: str
    outcome: str
    closed: bool
    entry_price: float
    exit_price: float
    avg_position: float
    pnl: float
    n_bars: int


@dataclass
class BacktestResult:
    signal_name: str
    notional: float
    equity: pd.Series            # cumulative PnL over time
    daily_pnl: pd.Series         # daily PnL
    trades: List[TradeResult]
    stats: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.stats = _compute_stats(self)


def run(
    signal: Signal,
    notional: float = 100.0,
    yes_only: bool = True,
) -> BacktestResult:
    """
    Run a single signal over all available markets.

    Args:
        signal:   Signal instance to evaluate.
        notional: Dollar amount allocated per market.
        yes_only: If True, only trade the Yes (token_index=0) side.

    Returns:
        BacktestResult with equity curve, per-trade breakdown, and stats.
    """
    prices_df, markets_df = load_data(yes_only=yes_only)
    meta = markets_df.set_index("token_id")

    all_pnl: List[pd.Series] = []
    trades: List[TradeResult] = []

    for token_id, grp in prices_df.groupby("token_id"):
        if token_id not in meta.index:
            continue

        info = meta.loc[token_id]
        history = grp[["timestamp", "price"]].sort_values("timestamp").copy()

        # Generate positions (signal fires at bar T, position applied T+1)
        raw_signal = signal.generate(history)
        positions = raw_signal.reindex(
            history.set_index("timestamp").index
        ).fillna(0.0).shift(1).fillna(0.0)

        prices = history.set_index("timestamp")["price"]

        # Mark-to-market PnL per bar: position * price_change * notional
        price_chg = prices.diff().fillna(0.0)
        bar_pnl = positions * price_chg * notional
        all_pnl.append(bar_pnl)

        # Per-market trade summary
        active = positions[positions != 0]
        if len(active) == 0:
            continue

        entry_price = prices.iloc[0]
        exit_price = prices.iloc[-1]
        total_pnl = bar_pnl.sum()

        trades.append(TradeResult(
            token_id=token_id,
            slug=info["slug"],
            question=info["question"],
            outcome=info["outcome"],
            closed=bool(info["closed"]),
            entry_price=entry_price,
            exit_price=exit_price,
            avg_position=active.mean(),
            pnl=total_pnl,
            n_bars=len(active),
        ))

    if not all_pnl:
        raise ValueError("No PnL data generated — check that prices.parquet has data.")

    # Combine across markets, align on common time index
    combined = pd.concat(all_pnl, axis=1).sort_index().fillna(0.0)
    total_pnl_series = combined.sum(axis=1)

    # Resample to daily for cleaner equity curve
    daily = total_pnl_series.resample("1D").sum()
    equity = daily.cumsum()

    return BacktestResult(
        signal_name=signal.name,
        notional=notional,
        equity=equity,
        daily_pnl=daily,
        trades=trades,
    )


def _compute_stats(result: BacktestResult) -> Dict:
    eq = result.equity
    daily = result.daily_pnl
    trades = result.trades

    total_return = eq.iloc[-1] if len(eq) else 0.0
    n_days = len(daily)

    # Sharpe (annualised, assuming hourly compounding daily)
    if daily.std() > 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = eq.cummax()
    drawdown = eq - running_max
    max_dd = drawdown.min()

    # Trade stats
    pnls = [t.pnl for t in trades]
    n_trades = len(pnls)
    win_rate = sum(p > 0 for p in pnls) / n_trades if n_trades else 0.0
    avg_pnl = np.mean(pnls) if pnls else 0.0
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0

    return {
        "total_pnl": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_pnl_per_market": avg_pnl,
        "best_market": best,
        "worst_market": worst,
        "n_days": n_days,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(result: BacktestResult, save_path: Optional[str] = None):
    """Plot a 6-panel dashboard for a single BacktestResult."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Polymarket Backtest — Signal: {result.signal_name}  "
        f"(notional ${result.notional:,.0f}/market)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :2])
    result.equity.plot(ax=ax1, color="steelblue", linewidth=1.5)
    ax1.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax1.set_title("Cumulative PnL ($)")
    ax1.set_ylabel("$")

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :2])
    running_max = result.equity.cummax()
    drawdown = result.equity - running_max
    drawdown.plot(ax=ax2, color="firebrick", linewidth=1.2)
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="firebrick")
    ax2.set_title("Drawdown ($)")
    ax2.set_ylabel("$")

    # 3. Per-market PnL bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    if result.trades:
        trade_df = pd.DataFrame([
            {"label": t.slug[:25], "pnl": t.pnl}
            for t in sorted(result.trades, key=lambda x: x.pnl)
        ])
        colors = ["firebrick" if p < 0 else "steelblue" for p in trade_df["pnl"]]
        ax3.barh(trade_df["label"], trade_df["pnl"], color=colors)
        ax3.axvline(0, color="black", linewidth=0.7)
        ax3.set_title("PnL per Market ($)")
        ax3.set_xlabel("$")
        ax3.tick_params(axis="y", labelsize=6)

    # 4. Return distribution
    ax4 = fig.add_subplot(gs[1, 2])
    if result.daily_pnl.std() > 0:
        result.daily_pnl.plot.hist(ax=ax4, bins=30, color="steelblue", edgecolor="white")
        ax4.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax4.axvline(
            result.daily_pnl.mean(), color="orange",
            linewidth=1.2, linestyle="--", label="mean",
        )
        ax4.legend(fontsize=8)
    ax4.set_title("Daily PnL Distribution")
    ax4.set_xlabel("$")

    # Stats box (lower-right corner of ax1)
    s = result.stats
    stats_text = (
        f"Total PnL:    ${s['total_pnl']:>8.2f}\n"
        f"Sharpe:       {s['sharpe']:>8.2f}\n"
        f"Max DD:       ${s['max_drawdown']:>8.2f}\n"
        f"Win rate:     {s['win_rate']:>8.1%}\n"
        f"Markets:      {s['n_trades']:>8d}\n"
        f"Avg PnL:      ${s['avg_pnl_per_market']:>8.2f}\n"
        f"Best:         ${s['best_market']:>8.2f}\n"
        f"Worst:        ${s['worst_market']:>8.2f}"
    )
    ax1.text(
        0.02, 0.97, stats_text,
        transform=ax1.transAxes,
        fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    return fig


def plot_compare(results: List[BacktestResult], save_path: Optional[str] = None):
    """Overlay equity curves for multiple signals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Signal Comparison — Polymarket Backtest", fontsize=13, fontweight="bold")

    # Equity curves
    ax = axes[0]
    for r in results:
        r.equity.plot(ax=ax, label=r.signal_name, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Cumulative PnL ($)")
    ax.legend(fontsize=9)
    ax.set_ylabel("$")

    # Stats bar chart
    ax2 = axes[1]
    names = [r.signal_name for r in results]
    total_pnls = [r.stats["total_pnl"] for r in results]
    colors = ["steelblue" if p >= 0 else "firebrick" for p in total_pnls]
    bars = ax2.bar(names, total_pnls, color=colors)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_title("Total PnL by Signal ($)")
    ax2.set_ylabel("$")
    ax2.tick_params(axis="x", rotation=15)
    for bar, pnl in zip(bars, total_pnls):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(total_pnls) - min(total_pnls)) * 0.01,
            f"${pnl:.1f}", ha="center", va="bottom", fontsize=8,
        )

    # Print stats table
    print(f"\n{'Signal':<20} {'PnL':>10} {'Sharpe':>8} {'MaxDD':>10} "
          f"{'WinRate':>9} {'Markets':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.stats["total_pnl"], reverse=True):
        s = r.stats
        print(
            f"{r.signal_name:<20} "
            f"${s['total_pnl']:>9.2f} "
            f"{s['sharpe']:>8.2f} "
            f"${s['max_drawdown']:>9.2f} "
            f"{s['win_rate']:>8.1%} "
            f"{s['n_trades']:>8d}"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {save_path}")
    else:
        plt.show()

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polymarket signal backtester")
    parser.add_argument(
        "--signal", default="all",
        help="Signal name or 'all' (default: all). "
             f"Options: {', '.join(ALL_SIGNALS)}",
    )
    parser.add_argument(
        "--notional", type=float, default=100.0,
        help="Dollar notional per market (default: 100)",
    )
    parser.add_argument(
        "--all-outcomes", action="store_true",
        help="Include No tokens as well as Yes",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save plot to this path instead of displaying",
    )
    args = parser.parse_args()

    yes_only = not args.all_outcomes

    if args.signal == "all":
        print("Running all signals…")
        results = []
        for name, sig in ALL_SIGNALS.items():
            print(f"  {name}…", end=" ", flush=True)
            r = run(sig, notional=args.notional, yes_only=yes_only)
            results.append(r)
            print(f"PnL=${r.stats['total_pnl']:.2f}")
        plot_compare(results, save_path=args.save)
    else:
        if args.signal not in ALL_SIGNALS:
            raise ValueError(
                f"Unknown signal '{args.signal}'. "
                f"Choose from: {', '.join(ALL_SIGNALS)}"
            )
        sig = ALL_SIGNALS[args.signal]
        print(f"Running {sig.name}…")
        r = run(sig, notional=args.notional, yes_only=yes_only)
        # Print per-market breakdown
        print(f"\n{'Slug':<45} {'Closed':>7} {'Entry':>7} "
              f"{'Exit':>7} {'AvgPos':>7} {'PnL':>8}")
        print("-" * 85)
        for t in sorted(r.trades, key=lambda x: x.pnl, reverse=True):
            print(
                f"{t.slug[:44]:<45} "
                f"{'Y' if t.closed else 'N':>7} "
                f"{t.entry_price:>7.3f} "
                f"{t.exit_price:>7.3f} "
                f"{t.avg_position:>7.2f} "
                f"${t.pnl:>7.2f}"
            )
        print("\nStats:")
        for k, v in r.stats.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:.4f}")
            else:
                print(f"  {k:<25} {v}")
        plot(r, save_path=args.save)


if __name__ == "__main__":
    main()

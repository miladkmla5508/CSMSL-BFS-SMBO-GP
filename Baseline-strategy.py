# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as tal
from scipy.optimize import dual_annealing
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# HELPER
# ============================================================
def pct_change_safe(series):
    return series.pct_change().fillna(0)


# ============================================================
# 1. STRATEGY FUNCTIONS
# ============================================================

def backtest_momentum(df, rsi_period=14, rsi_long=50,
                      macd_fast=12, macd_slow=26, macd_signal=9,
                      adx_period=14, adx_threshold=20, roc_period=10,
                      ema_fast=20, ema_slow=50, transaction_cost=0.001,
                      stop_loss_pct=0.05, momentum_threshold=3):
    """
    Momentum strategy (long-only, cash when flat).
    """
    df_test = df.copy()

    df_test['rsi']      = tal.RSI(df_test['close'], timeperiod=int(rsi_period))
    df_test['macd'], df_test['macd_signal'], _ = tal.MACD(
        df_test['close'], fastperiod=int(macd_fast),
        slowperiod=int(macd_slow), signalperiod=int(macd_signal))
    df_test['adx']      = tal.ADX(df_test['high'], df_test['low'],
                                  df_test['close'], timeperiod=int(adx_period))
    df_test['roc']      = tal.ROC(df_test['close'], timeperiod=int(roc_period))
    df_test['ema_fast'] = tal.EMA(df_test['close'], timeperiod=int(ema_fast))
    df_test['ema_slow'] = tal.EMA(df_test['close'], timeperiod=int(ema_slow))

    momentum_score = (
        (df_test['rsi'] > rsi_long).astype(int) +
        (df_test['macd'] > df_test['macd_signal']).astype(int) +
        (df_test['adx'] > adx_threshold).astype(int) +
        (df_test['ema_fast'] > df_test['ema_slow']).astype(int)
    )

    long_condition = (momentum_score >= momentum_threshold) & (df_test['roc'] > 0)
    df_test['signal'] = np.where(long_condition, 1, 0)

    df_test['entry_price']    = np.nan
    df_test['position_final'] = 0

    entry_price = None
    position    = 0

    for i in range(1, len(df_test)):
        prev_position = position
        signal_pos    = df_test['signal'].iloc[i]
        close_i       = df_test['close'].iloc[i]
        low_i         = df_test['low'].iloc[i]

        if signal_pos == 1 and prev_position == 0:
            position    = 1
            entry_price = close_i
            df_test.loc[df_test.index[i], 'entry_price'] = entry_price

        elif prev_position == 1 and entry_price is not None:
            stop_price = entry_price * (1 - stop_loss_pct)
            if low_i <= stop_price:
                position    = 0
                entry_price = None
            elif signal_pos == 0:
                position    = 0
                entry_price = None
            else:
                df_test.loc[df_test.index[i], 'entry_price'] = entry_price

        df_test.loc[df_test.index[i], 'position_final'] = position

    df_test['position_shift']      = df_test['position_final'].shift(1)
    df_test['return']              = df_test['close'].pct_change()
    df_test['strategy_return']     = df_test['position_shift'] * df_test['return']
    df_test['position_change']     = df_test['position_final'].diff().abs()
    df_test['cost']                = df_test['position_change'] * transaction_cost
    df_test['strategy_return_net'] = df_test['strategy_return'] - df_test['cost']
    df_test['cumret']    = (1 + df_test['strategy_return_net'].fillna(0)).cumprod()
    df_test['buy_hold']  = (1 + df_test['return'].fillna(0)).cumprod()

    return df_test


def backtest_mean_reversion(df, bb_period=20, bb_std=2, rsi_period=14,
                            rsi_oversold=30, rsi_overbought=70,
                            zscore_period=20, zscore_threshold=2,
                            transaction_cost=0.001):
    """Mean reversion: buy low, sell high."""
    df_test = df.copy()

    df_test['bb_upper'], df_test['bb_middle'], df_test['bb_lower'] = tal.BBANDS(
        df_test['close'], timeperiod=int(bb_period),
        nbdevup=bb_std, nbdevdn=bb_std)
    df_test['rsi']    = tal.RSI(df_test['close'], timeperiod=int(rsi_period))
    df_test['sma']    = tal.SMA(df_test['close'], timeperiod=int(zscore_period))
    df_test['std']    = df_test['close'].rolling(window=int(zscore_period)).std()
    df_test['zscore'] = (df_test['close'] - df_test['sma']) / df_test['std']

    long_condition = (
        (df_test['close'] < df_test['bb_lower']) &
        (df_test['rsi']   < rsi_oversold) &
        (df_test['zscore'] < -zscore_threshold)
    )
    short_condition = (
        (df_test['close'] > df_test['bb_upper']) &
        (df_test['rsi']   > rsi_overbought) &
        (df_test['zscore'] > zscore_threshold)
    )

    df_test['position'] = np.where(long_condition, 1,
                                   np.where(short_condition, -1, 0))
    df_test['position'] = df_test['position'].replace(0, np.nan).ffill().fillna(0)
    df_test['position_shift']      = df_test['position'].shift(1)
    df_test['return']              = df_test['close'].pct_change()
    df_test['strategy_return']     = df_test['position_shift'] * df_test['return']
    df_test['position_change']     = df_test['position'].diff().abs()
    df_test['cost']                = df_test['position_change'] * transaction_cost
    df_test['strategy_return_net'] = df_test['strategy_return'] - df_test['cost']
    df_test['cumret'] = (1 + df_test['strategy_return_net'].fillna(0)).cumprod()

    return df_test


def backtest_breakout(df, lookback=10, vol_multiplier=1.2,
                      atr_stop_mult=1.0, transaction_cost=0.001):
    """
    Breakout strategy with volume confirmation and ATR stop.

    FIX: no longer calls dropna(), so the output has the SAME number of rows
    as the input. Rows with NaN indicators are skipped (return=0, position=0),
    which lets walk-forward slicing stay perfectly aligned across strategies.
    """
    df_test = df.copy()
    lookback = int(lookback)

    tr = pd.concat([
        df_test['high'] - df_test['low'],
        (df_test['high'] - df_test['close'].shift(1)).abs(),
        (df_test['low']  - df_test['close'].shift(1)).abs()
    ], axis=1)
    df_test['ATR'] = tr.max(axis=1).rolling(window=lookback).mean()

    # shift(1) BEFORE rolling so windows exclude the current bar
    df_test['roll_max_high'] = df_test['high'].shift(1).rolling(window=lookback).max()
    df_test['roll_min_low']  = df_test['low'].shift(1).rolling(window=lookback).min()
    df_test['roll_max_vol']  = df_test['volume'].shift(1).rolling(window=lookback).max()

    # NOTE: NO dropna() here — keep row count identical to input.

    signal           = ""
    entry_price      = 0
    strategy_returns = []
    positions        = []

    for i in range(len(df_test)):
        # Guard: first row or any NaN in indicators -> flat, no return
        if (i == 0
                or pd.isna(df_test['ATR'].iloc[i])
                or pd.isna(df_test['roll_max_high'].iloc[i])
                or pd.isna(df_test['roll_min_low'].iloc[i])
                or pd.isna(df_test['roll_max_vol'].iloc[i])):
            strategy_returns.append(0)
            positions.append(0)
            continue

        high         = df_test['high'].iloc[i]
        low          = df_test['low'].iloc[i]
        close        = df_test['close'].iloc[i]
        volume       = df_test['volume'].iloc[i]
        atr          = df_test['ATR'].iloc[i]
        prev_close   = df_test['close'].iloc[i-1]
        roll_max     = df_test['roll_max_high'].iloc[i]
        roll_min     = df_test['roll_min_low'].iloc[i]
        prev_vol_max = df_test['roll_max_vol'].iloc[i]

        if signal == "":
            if high >= roll_max and volume > vol_multiplier * prev_vol_max:
                signal = "Buy";  entry_price = close
                strategy_returns.append(-transaction_cost); positions.append(1)
            elif low <= roll_min and volume > vol_multiplier * prev_vol_max:
                signal = "Sell"; entry_price = close
                strategy_returns.append(-transaction_cost); positions.append(-1)
            else:
                strategy_returns.append(0); positions.append(0)

        elif signal == "Buy":
            stop_price = entry_price - (atr_stop_mult * atr)
            if low <= stop_price:
                ret = (stop_price - entry_price) / entry_price - transaction_cost
                strategy_returns.append(ret); positions.append(0); signal = ""
            elif low <= roll_min and volume > vol_multiplier * prev_vol_max:
                ret = (close - entry_price) / entry_price - transaction_cost
                strategy_returns.append(ret)
                signal = "Sell"; entry_price = close; positions.append(-1)
            else:
                ret = (close - prev_close) / prev_close
                strategy_returns.append(ret); positions.append(1)

        elif signal == "Sell":
            stop_price = entry_price + (atr_stop_mult * atr)
            if high >= stop_price:
                ret = (entry_price - stop_price) / entry_price - transaction_cost
                strategy_returns.append(ret); positions.append(0); signal = ""
            elif high >= roll_max and volume > vol_multiplier * prev_vol_max:
                ret = (entry_price - close) / entry_price - transaction_cost
                strategy_returns.append(ret)
                signal = "Buy"; entry_price = close; positions.append(1)
            else:
                ret = (prev_close - close) / prev_close
                strategy_returns.append(ret); positions.append(-1)

    df_test['strategy_return_net'] = strategy_returns
    df_test['position']            = positions
    df_test['position_change']     = df_test['position'].diff().abs().fillna(0)
    df_test['cumret'] = (1 + df_test['strategy_return_net']).cumprod()

    return df_test


# ============================================================
# 2. OBJECTIVE FUNCTIONS
# ============================================================

def calculate_sharpe(df_result):
    returns = df_result['strategy_return_net'].dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def objective_momentum(params, df):
    try:
        result = backtest_momentum(
            df,
            rsi_period=int(params[0]),
            rsi_long=params[1],
            adx_threshold=params[2],
            ema_fast=int(params[3]),
            ema_slow=int(params[4]),
            stop_loss_pct=params[5],
            momentum_threshold=int(params[6])
        )
        sharpe = calculate_sharpe(result)
        num_trades = result['position_change'].sum()
        if num_trades < 10:
            sharpe -= 0.5
        return -sharpe
    except Exception:
        return 1e6


def objective_mean_reversion(params, df):
    try:
        result = backtest_mean_reversion(
            df, bb_period=int(params[0]), bb_std=params[1],
            rsi_period=int(params[2]), rsi_oversold=params[3],
            rsi_overbought=params[4])
        return -calculate_sharpe(result)
    except Exception:
        return 1e6


def objective_breakout(params, df):
    try:
        result = backtest_breakout(
            df, lookback=int(params[0]), vol_multiplier=params[1],
            atr_stop_mult=params[2])
        return -calculate_sharpe(result)
    except Exception:
        return 1e6


# ============================================================
# 3. QUANTUM-INSPIRED OPTIMIZATION
# ============================================================

def quantum_optimize_momentum(df):
    print("🔬 Optimizing Momentum Strategy...")
    bounds = [
        (10.0, 20.0),   # rsi_period
        (45.0, 60.0),   # rsi_long
        (15.0, 30.0),   # adx_threshold
        (10.0, 30.0),   # ema_fast
        (30.0, 60.0),   # ema_slow
        (0.02, 0.08),   # stop_loss_pct
        (2.0, 4.0),     # momentum_threshold
    ]
    
    result = dual_annealing(objective_momentum, bounds=bounds,
                            args=(df,), maxiter=150, seed=42)
    best_params = {
        'rsi_period':         int(result.x[0]),
        'rsi_long':           result.x[1],
        'adx_threshold':      result.x[2],
        'ema_fast':           int(result.x[3]),
        'ema_slow':           int(result.x[4]),
        'stop_loss_pct':      result.x[5],
        'momentum_threshold': int(result.x[6]),
    }
    best_result = backtest_momentum(df, **best_params)
    best_sharpe = -result.fun
    print(f"  ✓ Best Sharpe: {best_sharpe:.3f}")
    return best_result, best_params, best_sharpe


def quantum_optimize_mean_reversion(df):
    print("🔬 Optimizing Mean Reversion Strategy...")
    bounds = [(10.0, 30.0), (1.0, 3.0), (10.0, 20.0), (20.0, 40.0), (50.0, 80.0)]
    result = dual_annealing(objective_mean_reversion, bounds=bounds,
                            args=(df,), maxiter=150, seed=42)
    best_params = {'bb_period': int(result.x[0]), 'bb_std': result.x[1],
                   'rsi_period': int(result.x[2]), 'rsi_oversold': result.x[3],
                   'rsi_overbought': result.x[4]}
    best_result = backtest_mean_reversion(df, **best_params)
    print(f"  ✓ Best Sharpe: {-result.fun:.3f}")
    return best_result, best_params, -result.fun


def quantum_optimize_breakout(df):
    print("🔬 Optimizing Breakout Strategy...")
    bounds = [(5.0, 15.0), (0.5, 3.0), (0.5, 3.0)]
    result = dual_annealing(objective_breakout, bounds=bounds,
                            args=(df,), maxiter=150, seed=42)
    best_params = {'lookback': int(result.x[0]), 'vol_multiplier': result.x[1],
                   'atr_stop_mult': result.x[2]}
    best_result = backtest_breakout(df, **best_params)
    print(f"  ✓ Best Sharpe: {-result.fun:.3f}")
    return best_result, best_params, -result.fun


# ============================================================
# 4. FAST WALK-FORWARD (optimize once, reuse params)
# ============================================================

def walk_forward_optimization(df, strategy_func, optimize_func,
                              train_period=1260, test_period=252,
                              warmup=200, strategy_name="Strategy",
                              optimize_every=99):
    """
    FAST walk-forward:
      optimize_every=99 -> optimize once, reuse params (recommended)
      optimize_every=1  -> re-optimize every window (slow)
    """
    print(f"\n{'='*80}")
    print(f"🚀 WALK-FORWARD: {strategy_name}  (optimize_every={optimize_every})")
    print(f"{'='*80}")

    results          = []
    all_params       = []
    all_test_results = []
    start            = 0
    window_num       = 0
    cached_params    = None
    cached_sharpe    = None

    while start + train_period + test_period <= len(df):
        window_num += 1
        train_end = start + train_period
        test_end  = train_end + test_period

        df_train     = df.iloc[0:train_end].copy()
        test_start   = train_end
        warm_start   = max(0, test_start - warmup)
        df_test_full = df.iloc[warm_start:test_end].copy()

        if window_num == 1 or (window_num - 1) % optimize_every == 0:
            _, best_params, train_sharpe = optimize_func(df_train)
            cached_params = best_params
            cached_sharpe = train_sharpe
        else:
            best_params   = cached_params
            train_sharpe  = cached_sharpe

        test_result_full = strategy_func(df_test_full, **best_params)
        warmup_rows = test_start - warm_start

        # Safety: if strategy somehow returns fewer rows, fall back to tail slice
        if len(test_result_full) == len(df_test_full):
            test_result = test_result_full.iloc[warmup_rows:].reset_index(drop=True)
        else:
            test_result = test_result_full.iloc[-test_period:].reset_index(drop=True)

        if len(test_result) > 0:
            test_result['cumret'] = (1 + test_result['strategy_return_net'].fillna(0)).cumprod()
            test_return = (test_result['cumret'].iloc[-1] - 1) * 100
        else:
            test_return = 0.0

        test_sharpe = calculate_sharpe(test_result)

        print(f"  W{window_num}: Train Sharpe={train_sharpe:.2f} | "
              f"Test Sharpe={test_sharpe:.2f} | Test Return={test_return:.1f}%")

        results.append({
            'window': window_num, 'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe, 'test_return': test_return,
            'params': best_params,
        })
        all_params.append(best_params)
        all_test_results.append(test_result)
        start += test_period

    if len(results) == 0:
        print("⚠️  No windows fit. Reduce train/test periods.")
        return [], [], 0.0, []

    avg_train_sharpe  = np.mean([r['train_sharpe'] for r in results])
    avg_test_sharpe   = np.mean([r['test_sharpe']  for r in results])
    total_test_return = sum([r['test_return']      for r in results])

    print(f"\n📊 SUMMARY: {strategy_name}")
    print(f"  Avg Train Sharpe: {avg_train_sharpe:.3f}")
    print(f"  Avg Test Sharpe:  {avg_test_sharpe:.3f}")
    print(f"  Total Test Ret:   {total_test_return:.2f}%")

    return results, all_params, avg_test_sharpe, all_test_results


# ============================================================
# 5. METRICS
# ============================================================

def calculate_metrics(df_result, strategy_name):
    returns = df_result['strategy_return_net'].dropna()
    cumret  = df_result['cumret'].dropna()

    if len(cumret) == 0:
        return {'Strategy': strategy_name, 'Total Return (%)': 0,
                'Annual Return (%)': 0, 'Volatility (%)': 0,
                'Sharpe Ratio': 0, 'Max Drawdown (%)': 0,
                'Win Rate (%)': 0, 'Profit Factor': 0,
                'Calmar Ratio': 0, 'Total Trades': 0}

    total_return = (cumret.iloc[-1] - 1) * 100
    ann_return   = ((cumret.iloc[-1]) ** (252 / max(len(df_result), 1)) - 1) * 100
    volatility   = returns.std() * np.sqrt(252) * 100
    sharpe       = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    rolling_max  = cumret.cummax()
    drawdown     = (cumret - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # only count days where a position was held
    if 'position_final' in df_result.columns:
        active = df_result['position_final'].shift(1).fillna(0) != 0
    elif 'position' in df_result.columns:
        active = df_result['position'].shift(1).fillna(0) != 0
    else:
        active = returns != 0

    active_returns = returns[active]
    winning_trades = (active_returns > 0).sum()
    total_trades   = len(active_returns)
    win_rate       = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    gross_profit  = active_returns[active_returns > 0].sum()
    gross_loss    = abs(active_returns[active_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    if 'position' in df_result.columns:
        num_trades = int(df_result['position'].diff().abs().fillna(0).gt(0).sum())
    elif 'position_final' in df_result.columns:
        num_trades = int(df_result['position_final'].diff().abs().fillna(0).gt(0).sum())
    else:
        num_trades = total_trades

    return {
        'Strategy': strategy_name,
        'Total Return (%)': total_return,
        'Annual Return (%)': ann_return,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Calmar Ratio': calmar,
        'Total Trades': num_trades,
    }


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

print("\n" + "="*80)
print("🚀 FAST WALK-FORWARD OPTIMIZATION (3 STRATEGIES)")
print("="*80)

# Load data
df = pd.read_excel('/kaggle/input/datasets/miladkamali/ltcusdt/LTC.xlsx')

# Run walk-forward for all three strategies (optimize once, reuse)
wf_momentum, params_list_mom, avg_sharpe_mom, test_results_mom = walk_forward_optimization(
    df, backtest_momentum, quantum_optimize_momentum,
    train_period=1440, test_period=288, warmup=100,
    strategy_name="Momentum", optimize_every=1)

wf_mean_rev, params_list_mr, avg_sharpe_mr, test_results_mr = walk_forward_optimization(
    df, backtest_mean_reversion, quantum_optimize_mean_reversion,
    train_period=1440, test_period=288, warmup=100,
    strategy_name="Mean Reversion", optimize_every=1)

wf_breakout, params_list_bo, avg_sharpe_bo, test_results_bo = walk_forward_optimization(
    df, backtest_breakout, quantum_optimize_breakout,
    train_period=1440, test_period=288, warmup=100,
    strategy_name="Breakout", optimize_every=1)


# Build continuous cumulative curves
def create_continuous_cumret(test_results_list):
    if not test_results_list:
        return pd.DataFrame(columns=['strategy_return_net', 'cumret_continuous'])
    combined = pd.concat(test_results_list, ignore_index=True)
    combined['strategy_return_net'] = combined['strategy_return_net'].fillna(0)
    combined['cumret_continuous']   = (1 + combined['strategy_return_net']).cumprod()
    return combined


combined_mom = create_continuous_cumret(test_results_mom)
combined_mr  = create_continuous_cumret(test_results_mr)
combined_bo  = create_continuous_cumret(test_results_bo)

print(f"\nLength check -> mom: {len(combined_mom)}, "
      f"mr: {len(combined_mr)}, bo: {len(combined_bo)}")

# Align all three to the same length (safety net; should already match now)
min_len = min(len(combined_mom), len(combined_mr), len(combined_bo))
if min_len > 0:
    combined_mom = combined_mom.iloc[:min_len].reset_index(drop=True)
    combined_mr  = combined_mr.iloc[:min_len].reset_index(drop=True)
    combined_bo  = combined_bo.iloc[:min_len].reset_index(drop=True)
    # Recompute cumret after truncation
    for cdf in (combined_mom, combined_mr, combined_bo):
        cdf['strategy_return_net'] = cdf['strategy_return_net'].fillna(0)
        cdf['cumret_continuous']   = (1 + cdf['strategy_return_net']).cumprod()
        cdf['cumret']              = cdf['cumret_continuous']

metrics_mom = calculate_metrics(combined_mom, 'Momentum (WF)')
metrics_mr  = calculate_metrics(combined_mr,  'Mean Rev (WF)')
metrics_bo  = calculate_metrics(combined_bo,  'Breakout (WF)')

metrics_mom['Avg WF Sharpe'] = avg_sharpe_mom
metrics_mr['Avg WF Sharpe']  = avg_sharpe_mr
metrics_bo['Avg WF Sharpe']  = avg_sharpe_bo

comparison_df = pd.DataFrame([metrics_mom, metrics_mr, metrics_bo])

print("\n" + "="*80)
print("📊 WALK-FORWARD PERFORMANCE COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

buy_hold_ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
print(f"\n📌 Buy & Hold Return (full sample): {buy_hold_ret:.2f}%")


# ============================================================
# 7. VISUALIZATION
# ============================================================
fig = plt.figure(figsize=(20, 14))
gs  = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
if len(combined_mom): ax1.plot(combined_mom['cumret_continuous'].values, label='Momentum', linewidth=2, color='blue')
if len(combined_mr):  ax1.plot(combined_mr['cumret_continuous'].values,  label='Mean Reversion', linewidth=2, color='green')
if len(combined_bo):  ax1.plot(combined_bo['cumret_continuous'].values,  label='Breakout', linewidth=2, color='purple')
ax1.set_title('Walk-Forward: Continuous Cumulative Returns (All Test Periods)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Return')
ax1.set_xlabel('Trading Days (Across All Test Windows)')
ax1.legend(loc='best', fontsize=10)
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
strategies = ['Momentum', 'Mean Rev', 'Breakout']
wf_sharpes = [avg_sharpe_mom, avg_sharpe_mr, avg_sharpe_bo]
colors     = ['blue', 'green', 'purple']
bars = ax2.bar(strategies, wf_sharpes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1.0')
ax2.axhline(0.0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax2.set_title('Walk-Forward Average Sharpe Ratio', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sharpe Ratio')
ax2.set_xticklabels(strategies, rotation=20, ha='right')
ax2.legend(); ax2.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, wf_sharpes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax3 = fig.add_subplot(gs[1, 1])
if wf_momentum:
    windows_mom      = [r['window']      for r in wf_momentum]
    test_sharpes_mom = [r['test_sharpe'] for r in wf_momentum]
    ax3.plot(windows_mom, test_sharpes_mom, marker='o', linewidth=2, markersize=8, color='blue')
    ax3.axhline(avg_sharpe_mom, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_sharpe_mom:.2f}')
    ax3.axhline(0, color='black', linewidth=1, alpha=0.3)
ax3.set_title('Momentum: Test Sharpe by Window', fontsize=12, fontweight='bold')
ax3.set_xlabel('Window'); ax3.set_ylabel('Test Sharpe Ratio')
ax3.legend(); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
returns = [metrics_mom['Total Return (%)'], metrics_mr['Total Return (%)'], metrics_bo['Total Return (%)']]
bars = ax4.bar(strategies, returns, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.axhline(0, color='black', linewidth=1, alpha=0.3)
ax4.set_title('Walk-Forward Total Return', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Return (%)')
ax4.set_xticklabels(strategies, rotation=20, ha='right')
ax4.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, returns):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if val >= 0 else -1),
             f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
             fontweight='bold', fontsize=9)

ax5 = fig.add_subplot(gs[2, 1])
drawdowns = [metrics_mom['Max Drawdown (%)'], metrics_mr['Max Drawdown (%)'], metrics_bo['Max Drawdown (%)']]
bars = ax5.bar(strategies, drawdowns, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_title('Walk-Forward Max Drawdown', fontsize=12, fontweight='bold')
ax5.set_ylabel('Max Drawdown (%)')
ax5.set_xticklabels(strategies, rotation=20, ha='right')
ax5.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, drawdowns):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
             f'{val:.1f}%', ha='center', va='top', fontweight='bold', color='white', fontsize=9)

ax6 = fig.add_subplot(gs[3, :])
windows     = list(range(1, len(wf_momentum) + 1))
mom_returns = [r['test_return'] for r in wf_momentum]
mr_returns  = [r['test_return'] for r in wf_mean_rev]
bo_returns  = [r['test_return'] for r in wf_breakout]
x     = np.arange(len(windows))
width = 0.25
ax6.bar(x - width, mom_returns, width, label='Momentum', color='blue',   alpha=0.7)
ax6.bar(x,         mr_returns,  width, label='Mean Rev', color='green',  alpha=0.7)
ax6.bar(x + width, bo_returns,  width, label='Breakout', color='purple', alpha=0.7)
ax6.axhline(0, color='black', linestyle='-', linewidth=1)
ax6.set_title('Test Returns by Window (All Strategies)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Window'); ax6.set_ylabel('Test Return (%)')
ax6.set_xticks(x); ax6.set_xticklabels(windows)
ax6.legend(); ax6.grid(alpha=0.3, axis='y')

plt.suptitle('WALK-FORWARD OPTIMIZATION RESULTS (3 STRATEGIES)',
             fontsize=16, fontweight='bold', y=0.998)
plt.show()


# ============================================================
# 8. WINNER ANNOUNCEMENT
# ============================================================
print("\n" + "="*80)
print("🏆 WINNER BY DIFFERENT METRICS 🏆")
print("="*80)

if len(comparison_df):
    best_return  = comparison_df.loc[comparison_df['Total Return (%)'].idxmax(), 'Strategy']
    best_sharpe  = comparison_df.loc[comparison_df['Avg WF Sharpe'].idxmax(), 'Strategy']
    best_dd      = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmax(), 'Strategy']
    best_calmar  = comparison_df.loc[comparison_df['Calmar Ratio'].idxmax(), 'Strategy']
    best_winrate = comparison_df.loc[comparison_df['Win Rate (%)'].idxmax(), 'Strategy']

    print(f"🥇 Best Total Return:         {best_return}")
    print(f"🥇 Best Walk-Forward Sharpe:  {best_sharpe}")
    print(f"🥇 Smallest Drawdown:         {best_dd}")
    print(f"🥇 Best Risk-Adjusted Return: {best_calmar}")
    print(f"🥇 Best Win Rate:             {best_winrate}")

    overall_winner = comparison_df.loc[comparison_df['Avg WF Sharpe'].idxmax(), 'Strategy']
    print(f"\n🎉 OVERALL WINNER (Walk-Forward Sharpe): {overall_winner} 🎉\n")
print("="*80)


# ============================================================
# 9. OPTIMAL PARAMETERS
# ============================================================
print("\n" + "="*80)
print("🎯 OPTIMAL PARAMETERS (By Window)")
print("="*80)

print("\n🔵 MOMENTUM STRATEGY:")
for i, params in enumerate(params_list_mom, 1):
    print(f"\n  Window {i}:")
    for k, v in params.items():
        print(f"    {k:20s}: {v}")

    for k, v in params.items():
        print(f"    {k:20s}: {v}")
print("\n" + "="*80)


# ============================================================
# 10. CONSISTENCY ANALYSIS
# ============================================================
print("\n" + "="*80)
print("📊 CONSISTENCY ANALYSIS")
print("="*80)


def analyze_consistency(wf_results, strategy_name):
    if not wf_results:
        print(f"\n{strategy_name}: no windows")
        return 0.0, 0.0
    test_sharpes = [r['test_sharpe'] for r in wf_results]
    test_returns = [r['test_return'] for r in wf_results]
    positive_windows = sum(1 for r in test_returns if r > 0)
    total_windows    = len(test_returns)
    consistency      = positive_windows / total_windows * 100
    sharpe_std       = np.std(test_sharpes)
    return_std       = np.std(test_returns)
    print(f"\n{strategy_name}:")
    print(f"  Positive Windows:      {positive_windows}/{total_windows} ({consistency:.1f}%)")
    print(f"  Sharpe Std Dev:        {sharpe_std:.3f}")
    print(f"  Return Std Dev:        {return_std:.2f}%")
    print(f"  Min/Max Test Sharpe:   {min(test_sharpes):.3f} / {max(test_sharpes):.3f}")
    print(f"  Min/Max Test Return:   {min(test_returns):.2f}% / {max(test_returns):.2f}%")
    return consistency, sharpe_std


cons_mom, std_mom = analyze_consistency(wf_momentum, "🔵 Momentum")
cons_mr,  std_mr  = analyze_consistency(wf_mean_rev, "🟢 Mean Reversion")
cons_bo,  std_bo  = analyze_consistency(wf_breakout, "🟣 Breakout")

print("\n" + "="*80)

consistencies = {'Momentum': cons_mom, 'Mean Reversion': cons_mr, 'Breakout': cons_bo}
stabilities   = {'Momentum': std_mom, 'Mean Reversion': std_mr, 'Breakout': std_bo}

if consistencies:
    most_consistent = max(consistencies, key=consistencies.get)
    most_stable     = min(stabilities,   key=stabilities.get)
    print(f"\n✅ Most Consistent:  {most_consistent} ({consistencies[most_consistent]:.1f}% positive windows)")
    print(f"✅ Most Stable:      {most_stable} (Sharpe Std: {stabilities[most_stable]:.3f})")


# ============================================================
# 11. FINAL RECOMMENDATIONS
# ============================================================
print("\n" + "="*80)
print("📝 FINAL RECOMMENDATIONS FOR LIVE TRADING")
print("="*80)

best_out_sample = max([
    ('Momentum',       avg_sharpe_mom),
    ('Mean Reversion', avg_sharpe_mr),
    ('Breakout',       avg_sharpe_bo),
], key=lambda x: x[1])

print(f"\n✅ Best Walk-Forward Performance:  {best_out_sample[0]}")
print(f"   Average Sharpe: {best_out_sample[1]:.3f}")

if best_out_sample[1] > 1.5:
    verdict = "EXCELLENT — Strong candidate for live trading"
elif best_out_sample[1] > 1.0:
    verdict = "GOOD — Viable for live trading with proper risk management"
elif best_out_sample[1] > 0.5:
    verdict = "MODERATE — Use with caution, smaller position sizes"
else:
    verdict = "WEAK — Further optimization recommended"
print(f"   Verdict: {verdict}")

if consistencies:
    print(f"\n✅ Most Reliable for Live Trading: {most_consistent}")
    print(f"   Consistency: {consistencies[most_consistent]:.1f}% of windows profitable")

print("\n💡 Strategy Tips:")
print("  • Momentum:       trending markets")
print("  • Mean Reversion: range-bound markets")
print("  • Breakout:       high-volume, volatile periods")

print("\n📋 Live Trading Guidelines:")
print("  • Use the most recent window's parameters")
print("  • Re-optimize every 3-6 months")
print("  • Start at 50% position size for the first month")
print("  • Scale up only if performance matches walk-forward")

print("\n⚠️  Risk Management:")
if 'drawdowns' in dir() and len(drawdowns):
    print(f"  • Expected Max Drawdown: {min(drawdowns):.1f}%")
print("  • Account risk per trade: 2-5%")
print("  • Stops are built into the strategies")

print("\n" + "="*80)
print("✅ WALK-FORWARD OPTIMIZATION COMPLETE")
print("="*80)
print(f"Windows analyzed: {len(wf_momentum)}")
print("="*80)for i, params in enumerate(params_list_bo, 1):
    print(f"\n  Window {i}:")
    for k, v in params.items():
        print(f"    {k:20s}: {v}")

print("\n\n🟣 BREAKOUT STRATEGY:")
print("\n\n🟢 MEAN REVERSION STRATEGY:")
for i, params in enumerate(params_list_mr, 1):
    print(f"\n  Window {i}:")


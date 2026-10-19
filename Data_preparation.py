# ============================================================
# IMPORTS  (only what is actually used)
# ============================================================
from talib import (
    RSI, WILLR, MOM, ROCP, CCI, CMO, ROC, MFI,
    ULTOSC, ADOSC, ATR, MINUS_DM, PLUS_DM
)
import ta

# ============================================================
# BOLLINGER BANDS  (BB_P_6 .. BB_P_20)
# ============================================================
for w in range(6, 16):
    df[f'BB_P_{w}'] = ta.volatility.BollingerBands(
        close=df['close'], window=w, window_dev=1
    ).bollinger_pband()

for i, w in enumerate(range(6, 11), start=16):
    df[f'BB_P_{i}'] = ta.volatility.BollingerBands(
        close=df['close'], window=w, window_dev=2
    ).bollinger_pband()

# ============================================================
# DONCHIAN CHANNEL  (DC_P_6 .. DC_P_20)
# ============================================================
for w in range(6, 21):
    df[f'DC_P_{w}'] = ta.volatility.donchian_channel_pband(
        high=df['high'], low=df['low'],
        close=df['close'], window=w, offset=0
    )

# ============================================================
# KELTNER CHANNEL  (KC_P_6 .. KC_P_20)
# ============================================================
kc = 6
for atr_w in (6, 7, 8):
    for w in range(6, 11):
        df[f'KC_P_{kc}'] = ta.volatility.KeltnerChannel(
            high=df['high'], low=df['low'],
            close=df['close'], window=w,
            window_atr=atr_w, multiplier=2
        ).keltner_channel_pband()
        kc += 1

# ============================================================
# STOCHASTIC OSCILLATOR  (SLOWD_6 .. SLOWD_20)
# ============================================================
sd = 6
for smooth in (1, 2, 3):
    for w in range(6, 11):
        df[f'SLOWD_{sd}'] = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'],
            window=w, smooth_window=smooth
        ).stoch_signal()
        sd += 1

# ============================================================
# AROON  (AROON_DOWN_6..12, AROON_UP_6..12)
# ============================================================
for w in range(6, 13):
    a = ta.trend.AroonIndicator(high=df['high'], low=df['low'], window=w)
    df[f'AROON_DOWN_{w}'] = a.aroon_down()
    df[f'AROON_UP_{w}']   = a.aroon_up()

# ============================================================
# ADX  (ADX_NEG_6..12, ADX_POS_6..12)
# ============================================================
for w in range(6, 13):
    a = ta.trend.ADXIndicator(high=df['high'], low=df['low'],
                              close=df['close'], window=w)
    df[f'ADX_NEG_{w}'] = a.adx_neg()
    df[f'ADX_POS_{w}'] = a.adx_pos()

# ============================================================
# WILLIAMS %R  (WILLIAM_6 .. WILLIAM_20)
# ============================================================
for w in range(6, 21):
    df[f'WILLIAM_{w}'] = WILLR(df['high'], df['low'],
                               df['close'], timeperiod=w)

# ============================================================
# MOM, CMO, ROC, RSI, ROCP  (6 .. 20)
# ============================================================
for w in range(6, 21):
    df[f'MOM_{w}']  = MOM(df['close'],  timeperiod=w)
    df[f'CMO_{w}']  = CMO(df['close'],  timeperiod=w)
    df[f'ROC_{w}']  = ROC(df['close'],  timeperiod=w)
    df[f'RSI_{w}']  = RSI(df['close'],  timeperiod=w)
    df[f'ROCP_{w}'] = ROCP(df['close'], timeperiod=w)

# ============================================================
# AWESOME OSCILLATOR  (AwesomeOscillatorIndicator_6 .. _20)
# ============================================================
ao = 6
for w2 in (16, 17, 18):
    for w1 in range(6, 11):
        df[f'AwesomeOscillatorIndicator_{ao}'] = ta.momentum.awesome_oscillator(
            df['high'], df['low'], window1=w1, window2=w2)
        ao += 1

# ============================================================
# CCI  (CCI_6 .. CCI_20)
# ============================================================
for w in range(6, 21):
    df[f'CCI_{w}'] = CCI(df['high'], df['low'],
                         df['close'], timeperiod=w)

# ============================================================
# ULTOSC  (ULTOSC_6 .. ULTOSC_20)
# ============================================================
ult = 6
for tp1 in range(6, 9):
    for tp2, tp3 in zip(range(14, 19), range(26, 31)):
        df[f'ULTOSC_{ult}'] = ULTOSC(
            df['high'], df['low'], df['close'],
            timeperiod1=tp1, timeperiod2=tp2, timeperiod3=tp3)
        ult += 1

# ============================================================
# MFI  (MFI_6 .. MFI_20)
# ============================================================
for w in range(6, 21):
    df[f'MFI_{w}'] = MFI(df['close'], df['high'],
                         df['low'], df['volume'],
                         timeperiod=w)

# ============================================================
# MACD  (MACD_6 .. MACD_20)
# ============================================================
macd = 6
for slow, fast in ((20, 6), (21, 7), (22, 8), (23, 9)):
    for sign in (1, 2, 3, 4):
        if macd > 20: break
        df[f'MACD_{macd}'] = ta.trend.MACD(
            close=df['close'], window_slow=slow,
            window_fast=fast, window_sign=sign).macd_diff()
        macd += 1

# ============================================================
# PPO  (PPO_6 .. PPO_20)
# ============================================================
ppo = 6
for slow, fast in ((20, 6), (21, 7), (22, 8), (23, 9)):
    for sign in (1, 2, 3, 4):
        if ppo > 20: break
        df[f'PPO_{ppo}'] = ta.momentum.PercentagePriceOscillator(
            close=df['close'], window_slow=slow,
            window_fast=fast, window_sign=sign).ppo_hist()
        ppo += 1

# ============================================================
# CHAIKIN OSCILLATOR  (ChaikinOscillator_6 .. _20)
# ============================================================
ch = 6
for slow in (6, 7, 8, 9):
    for fast in (2, 3, 4, 5):
        if ch > 20: break
        df[f'ChaikinOscillator_{ch}'] = ADOSC(
            high=df['high'], low=df['low'],
            close=df['close'], volume=df['volume'],
            fastperiod=fast, slowperiod=slow)
        ch += 1

# ============================================================
# FORCE INDEX  (FI_6 .. FI_20)
# ============================================================
for w in range(6, 21):
    df[f'FI_{w}'] = ta.volume.ForceIndexIndicator(
        close=df['close'], volume=df['volume'],
        window=w).force_index()

# ============================================================
# TSI  (TSI_6 .. TSI_20)
# ============================================================
tsi = 6
for fast in (1, 2, 3):
    for slow in (2, 3, 4, 5, 6):
        df[f'TSI_{tsi}'] = ta.momentum.TSIIndicator(
            close=df['close'], window_slow=slow,
            window_fast=fast).tsi()
        tsi += 1

# ============================================================
# ULCER INDEX  (UI_6 .. UI_20)
# ============================================================
for w in range(6, 21):
    df[f'UI_{w}'] = ta.volatility.UlcerIndex(
        close=df['close'], window=w).ulcer_index()

# ============================================================
# ATR  (ATR_6 .. ATR_20)
# ============================================================
for w in range(6, 21):
    df[f'ATR_{w}'] = ATR(df['high'], df['low'],
                         df['close'], timeperiod=w)

# ============================================================
# MINUS_DM / PLUS_DM  (6 .. 20)
# ============================================================
for w in range(6, 21):
    df[f'MINUS_DM_{w}'] = MINUS_DM(df['high'], df['low'], timeperiod=w)
    df[f'PLUS_DM_{w}']  = PLUS_DM(df['high'], df['low'], timeperiod=w)

# ============================================================
# EASE OF MOVEMENT  (EOM_6 .. EOM_20)
# ============================================================
for w in range(6, 21):
    df[f'EOM_{w}'] = ta.volume.EaseOfMovementIndicator(
        high=df['high'], low=df['low'],
        volume=df['volume'], window=w).ease_of_movement()

# ============================================================
# FINAL: shift all indicator columns by -1 and drop NaN
# ============================================================
df.dropna(inplace=True)
df.loc[:, 'BB_P_6':'EOM_20'] = df.loc[:, 'BB_P_6':'EOM_20'].shift(-1)
df.dropna(inplace=True)

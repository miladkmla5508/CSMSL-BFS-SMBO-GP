import mplfinance as fplt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

df=pd.read_excel('/kaggle/input/datasets/miladkamali/solusdt/SOL.xlsx')
df.set_index('date',inplace=True)

# --- Zigzag Logic ---
label_series = df['LABEL'][:580].values
swing_indices = [0]

# Find swing points (where LABEL_TREND switches between 0 and 1)
for i in range(1, len(label_series)):
    if label_series[i] != label_series[i - 1]:
        swing_indices.append(i-1)

# Add last point to complete the zigzag line
if swing_indices[-1] != len(label_series) - 1:
    swing_indices.append(len(label_series) - 1)

swing_dates = df.index[swing_indices]
swing_prices = df['close'].iloc[swing_indices].values

# Initialize zigzag line array with NaNs
zigzag_line_arr = np.full(len(df[:580]), np.nan)
zigzag_dot_arr = np.full(len(df[:580]), np.nan)
for i, idx in enumerate(swing_indices):
    zigzag_dot_arr[idx] = swing_prices[i]

# Fill zigzag line with interpolated values
for i in range(len(swing_indices) - 1):
    start_idx = swing_indices[i]
    end_idx = swing_indices[i + 1]
    start_price = df['close'][:580].iloc[start_idx]
    end_price = df['close'][:580].iloc[end_idx]

    steps = end_idx - start_idx + 1
    interpolated = np.linspace(start_price, end_price, steps)
    zigzag_line_arr[start_idx:end_idx + 1] = interpolated


# Plotting components
green_circles = fplt.make_addplot(
    np.where(df['LABEL'][:580] == 1, 1, np.nan),
    type='scatter', marker='o', markersize=50, color='green', panel=1
)

red_circles = fplt.make_addplot(
    np.where(df['LABEL'][:580] == 0, 0, np.nan),
    type='scatter', marker='o', markersize=50, color='red', panel=1
)

# Zigzag line and dots
zigzag_line = fplt.make_addplot(zigzag_line_arr, panel=0, type='line',
                                markersize=100, color='black', alpha=1, linestyle="-")
zigzag_dot = fplt.make_addplot(zigzag_dot_arr, panel=0, type='scatter',
                               markersize=100, color='black', alpha=1)

close_plot = fplt.make_addplot(df['LABEL'][:580], panel=1, color='black', linestyle='-')

# Define vertical lines based on LABEL_TREND
correct_buy_dates = df.index[:580][df['LABEL'][:580] == 1].tolist()
correct_sell_dates = df.index[:580][df['LABEL'][:580] == 0].tolist()

vlines_dict = {
    'vlines': correct_buy_dates + correct_sell_dates,
    'linewidths': [2] * (len(correct_buy_dates) + len(correct_sell_dates)),
    'colors': ['green'] * len(correct_buy_dates) + ['red'] * len(correct_sell_dates),
    'linestyle': ['-'] * (len(correct_buy_dates) + len(correct_sell_dates)),
    'alpha': 0.5,
}

# Plotting
fig, axlist = fplt.plot(
    df[:580],
    type='candle',
    volume=False,
    update_width_config=dict(candle_width=0.7),
    addplot=[close_plot, green_circles, red_circles, zigzag_line, zigzag_dot],
    figratio=(120, 70),
    panel_ratios=(7, 2),
    figscale=2,
    tight_layout=True,
    style='binance',
    ylabel='Price ($)',
    main_panel=0,
    num_panels=2,
    vlines=vlines_dict,
    returnfig=True
)

# Long/Short annotations
long_positions = df.index[:1440][(df['LABEL'][:1440] == 1)].tolist()
short_positions = df.index[:1440][(df['LABEL'][:1440] == 0)].tolist()
long_prices = df['high'][:1440][df['LABEL'][:1440] == 1]
short_prices = df['low'][:1440][df['LABEL'][:1440] == 0]

ax = axlist[0]
for date, price in zip(long_positions, long_prices):
    ax.annotate('Long', xy=(date, price), xytext=(date, price + 0.01),
                textcoords='offset points', ha='center', fontsize=30, color='green',
                arrowprops=dict(facecolor='green', arrowstyle='->', lw=1.5))
for date, price in zip(short_positions, short_prices):
    ax.annotate('Short', xy=(date, price), xytext=(date, price - 0.01),
                textcoords='offset points', ha='center', fontsize=30, color='red',
                arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5))

# Custom Legend
custom_legend = [
    Line2D([0], [0], color='green', marker='o', markersize=25, label='Long', linestyle='None'),
    Line2D([0], [0], color='red', marker='o', markersize=25, label='Short', linestyle='None'),
    Line2D([0], [0], color='black', linestyle='-', linewidth=3, label='Swing body'),
    Line2D([0], [0], color='black', marker='o', markersize=25, label='Swing points', linestyle='None')
]
ax.legend(handles=custom_legend, loc='lower center', fontsize=30, frameon=True, ncol=4, title='')

# Title and axes tweaks
fig.suptitle('BTC/U.S. Dollar -- 1min', y=0.95, fontsize=30, x=0.85, style='oblique', fontfamily='sans-serif')
axlist[1].set_title('', fontsize=40, loc='center')
handles, labels = axlist[1].get_legend_handles_labels()
axlist[1].legend(handles, labels, fontsize=50, ncol=4, loc='center',
                 bbox_to_anchor=(50, 1), title='Trading Signals', title_fontsize=2)

# Axis ticks styling
for ax in axlist:
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_ylabel(ax.get_ylabel(), fontsize=30)
    ax.set_xlabel(ax.get_xlabel(), fontsize=30)

plt.show()
#fig.savefig(f"/content/labeling.tiff", bbox_inches='tight')
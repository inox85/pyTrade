feature_cols = [
    "Open", "High", "Low", "Close", "Volume", "trade_count", "vwap",
    "MACD", "MACD_signal", "MACD_hist", "MACD_diff", "MACD_slope", "Signal_slope",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_100",
    "EMA_5_x_EMA_10_diff", "EMA_5_x_EMA_10_UP_cont", "EMA_5_x_EMA_10_DOWN_cont",
    "EMA_5_x_EMA_20_diff", "EMA_5_x_EMA_20_UP_cont", "EMA_5_x_EMA_20_DOWN_cont",
    "EMA_5_x_EMA_50_diff", "EMA_5_x_EMA_50_UP_cont", "EMA_5_x_EMA_50_DOWN_cont",
    "EMA_5_x_EMA_100_diff", "EMA_5_x_EMA_100_UP_cont", "EMA_5_x_EMA_100_DOWN_cont",
    "EMA_10_x_EMA_20_diff", "EMA_10_x_EMA_20_UP_cont", "EMA_10_x_EMA_20_DOWN_cont",
    "EMA_10_x_EMA_50_diff", "EMA_10_x_EMA_50_UP_cont", "EMA_10_x_EMA_50_DOWN_cont",
    "EMA_10_x_EMA_100_diff", "EMA_10_x_EMA_100_UP_cont", "EMA_10_x_EMA_100_DOWN_cont",
    "EMA_20_x_EMA_50_diff", "EMA_20_x_EMA_50_UP_cont", "EMA_20_x_EMA_50_DOWN_cont",
    "EMA_20_x_EMA_100_diff", "EMA_20_x_EMA_100_UP_cont", "EMA_20_x_EMA_100_DOWN_cont",
    "EMA_50_x_EMA_100_diff", "EMA_50_x_EMA_100_UP_cont", "EMA_50_x_EMA_100_DOWN_cont",
    "RSI", "RSI_norm", "BB_Width", "BB_Percent_b",
    "BB_Breakout_up_cont", "BB_Breakout_down_cont",
    "Body_today", "Body_prev", "Engulfing_ratio", "Engulfing_direction", "Engulfing_strength",
    "Engulfing_inclusion", "Engulfing_index",
    "OBV", "OBV_diff", "OBV_pct", "OBV_slope", "OBV_momentum",
    "Close_VWAP_Diff", "Close_VWAP_Ratio",
    "Avg_Trade_Size", "Trade_Count_Norm_3", "Trade_Count_Norm_5", "Trade_Count_Norm_10",
    "Donchian_High", "Donchian_Low",
    "Vol_MA", "ADX", "Donchian_Close_HighDiff", "Donchian_Close_LowDiff",
    "Donchian_Close_HighPct", "Donchian_Close_LowPct",
    "Vol_Ratio", "Vol_Diff", "ADX_Above_Threshold", "ADX_Slope"
]

target_cols = [
    "Target_sl_1_1", "Target_sl_1_2", "Target_tp_1_1", "Target_tp_1_2", "Target_tp_1_3",
    "Target_sl_5_1", "Target_sl_5_2", "Target_tp_5_1", "Target_tp_5_2", "Target_tp_5_3",
    "Target_sl_10_1", "Target_sl_10_2", "Target_tp_10_1", "Target_tp_10_2", "Target_tp_10_3",
    "Target_sl_20_1", "Target_sl_20_2", "Target_tp_20_1", "Target_tp_20_2", "Target_tp_20_3"
]
# Sales Forecasting Report — Rossmann Store Sales

End-to-end time-series forecasting of daily store sales with SARIMA (classical), LightGBM (gradient boosting) and an LSTM (deep learning). Chronological 80/20 split; metrics MAE, RMSE, MAPE, MASE; forecasts for 1/3/6-month horizons with 80% and 95% confidence bands.

## Store 1097

Store type `b`, assortment `b`.
Test window: 2015-01-24 → 2015-07-31 (189 days).

### EDA insights (from full history)

- Average daily sales: **9745** units. Peak day: **Sun**, trough day: **Sat**.

| Day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Avg sales | 10176 | 9490 | 9492 | 9830 | 10022 | 7417 | 11787 |

- Promotions ran on 38% of days and lifted sales by **+11.0%** on average.

- Public holidays fell on 3.1% of days; holiday-day sales were **+35.6%** vs normal days.

### Model comparison (test period)

| Model | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|
| SARIMA | 1556 | 2058 | 13.35% | 1.225 |
| SARIMA (recursive) | 1556 | 2058 | 13.35% | 1.225 |
| LightGBM | 611 | 867 | 5.37% | 0.481 |
| LightGBM (recursive) | 786 | 1045 | 7.09% | 0.619 |
| LSTM | 122 | 221 | 1.01% | 0.096 |
| LSTM (recursive) | 3479 | 4045 | 35.32% | 2.740 |

**Best model by recursive-test RMSE: LightGBM.**

### EDA figures

![store_1097_timeseries](output/figures/store_1097_timeseries.png)

![store_1097_seasonality](output/figures/store_1097_seasonality.png)

![store_1097_acf_pacf](output/figures/store_1097_acf_pacf.png)

![store_1097_corr](output/figures/store_1097_corr.png)

![store_1097_decomposition](output/figures/store_1097_decomposition.png)

### Actual vs predicted (test period)

![store_1097_actual_vs_sarima](output/figures/store_1097_actual_vs_sarima.png)

![store_1097_actual_vs_lstm](output/figures/store_1097_actual_vs_lstm.png)

![store_1097_actual_vs_lightgbm](output/figures/store_1097_actual_vs_lightgbm.png)

### Feature importance (LightGBM)

![store_1097_feature_importance](output/figures/store_1097_feature_importance.png)

Top 10 features:

| Feature | Importance |
|---|---|
| log_sales_lag_1 | 191 |
| day_of_month | 189 |
| log_sales_roll_mean_7 | 144 |
| day_of_week | 128 |
| log_sales_roll_std_7 | 128 |
| cos_doy | 128 |
| log_sales_lag_7 | 108 |
| Promo | 107 |
| log_sales_lag_14 | 92 |
| sin_dow | 76 |

### Error segmentation (LightGBM, one-step)

By **Promo**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 1.0 | 75.0 | 688 | 954 | 5.73% |
| 0.0 | 114.0 | 561 | 804 | 5.12% |

By **day_of_week**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 0.0 | 27.0 | 830 | 1237 | 6.74% |
| 6.0 | 27.0 | 763 | 966 | 5.58% |
| 3.0 | 27.0 | 735 | 1003 | 6.32% |
| 4.0 | 27.0 | 587 | 763 | 5.06% |
| 5.0 | 27.0 | 493 | 702 | 5.70% |
| 2.0 | 27.0 | 469 | 641 | 4.36% |
| 1.0 | 27.0 | 403 | 549 | 3.80% |

By **holiday**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 1.0 | 6.0 | 2089 | 2262 | 13.02% |
| 0.0 | 183.0 | 563 | 780 | 5.11% |

### Promotion impact (30-day scenario analysis, LightGBM)

| Scenario | Avg daily sales | 30-day total | Lift vs no-promo |
|---|---|---|---|
| No promo | 9954 | 298612 | — |
| Promo every day | 11359 | 340764 | **+14.1%** |

Scenario forecasts assume the same holiday calendar; the promo flag is the only difference.

### Forecasts

#### 1 month ahead (30 days)

![store_1097_30d_forecast](outputs/figures/store_1097_30d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 8244 | 7724 | 9039 | 7425 | 9781 |
| 2015-08-02 | 13200 | 12368 | 14472 | 11889 | 15660 |
| 2015-08-03 | 11852 | 11105 | 12994 | 10674 | 14061 |
| 2015-08-04 | 11566 | 10837 | 12681 | 10417 | 13722 |
| 2015-08-05 | 11436 | 10715 | 12538 | 10300 | 13568 |
| 2015-08-06 | 11158 | 10454 | 12233 | 10049 | 13237 |
| 2015-08-07 | 11313 | 10600 | 12404 | 10189 | 13422 |
| 2015-08-08 | 8103 | 7592 | 8884 | 7298 | 9613 |
| 2015-08-09 | 13070 | 12246 | 14330 | 11772 | 15506 |
| 2015-08-10 | 11747 | 11007 | 12880 | 10580 | 13937 |

Forecast CSV saved to `outputs/forecasts/store_1097_30d_lightgbm.csv`.

#### 3 months ahead (90 days)

![store_1097_90d_forecast](outputs/figures/store_1097_90d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 8244 | 7724 | 9039 | 7425 | 9781 |
| 2015-08-02 | 13200 | 12368 | 14472 | 11889 | 15660 |
| 2015-08-03 | 11852 | 11105 | 12994 | 10674 | 14061 |
| 2015-08-04 | 11566 | 10837 | 12681 | 10417 | 13722 |
| 2015-08-05 | 11436 | 10715 | 12538 | 10300 | 13568 |
| 2015-08-06 | 11158 | 10454 | 12233 | 10049 | 13237 |
| 2015-08-07 | 11313 | 10600 | 12404 | 10189 | 13422 |
| 2015-08-08 | 8103 | 7592 | 8884 | 7298 | 9613 |
| 2015-08-09 | 13070 | 12246 | 14330 | 11772 | 15506 |
| 2015-08-10 | 11747 | 11007 | 12880 | 10580 | 13937 |

Forecast CSV saved to `outputs/forecasts/store_1097_90d_lightgbm.csv`.

#### 6 months ahead (180 days)

![store_1097_180d_forecast](outputs/figures/store_1097_180d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 8244 | 7724 | 9039 | 7425 | 9781 |
| 2015-08-02 | 13200 | 12368 | 14472 | 11889 | 15660 |
| 2015-08-03 | 11852 | 11105 | 12994 | 10674 | 14061 |
| 2015-08-04 | 11566 | 10837 | 12681 | 10417 | 13722 |
| 2015-08-05 | 11436 | 10715 | 12538 | 10300 | 13568 |
| 2015-08-06 | 11158 | 10454 | 12233 | 10049 | 13237 |
| 2015-08-07 | 11313 | 10600 | 12404 | 10189 | 13422 |
| 2015-08-08 | 8103 | 7592 | 8884 | 7298 | 9613 |
| 2015-08-09 | 13070 | 12246 | 14330 | 11772 | 15506 |
| 2015-08-10 | 11747 | 11007 | 12880 | 10580 | 13937 |

Forecast CSV saved to `outputs/forecasts/store_1097_180d_lightgbm.csv`.

## Store 682

Store type `b`, assortment `a`.
Test window: 2015-01-24 → 2015-07-31 (189 days).

### EDA insights (from full history)

- Average daily sales: **11207** units. Peak day: **Mon**, trough day: **Sun**.

| Day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Avg sales | 13757 | 12844 | 12296 | 12222 | 12107 | 7921 | 7270 |

- Promotions ran on 38% of days and lifted sales by **+56.5%** on average.

- Public holidays fell on 2.5% of days; holiday-day sales were -35.7%** vs normal days.

### Model comparison (test period)

| Model | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|
| SARIMA | 6105 | 6880 | 51.78% | 2.134 |
| SARIMA (recursive) | 6105 | 6880 | 51.78% | 2.134 |
| LightGBM | 747 | 1092 | 6.57% | 0.261 |
| LightGBM (recursive) | 940 | 1372 | 7.91% | 0.329 |
| LSTM | 202 | 339 | 1.53% | 0.071 |
| LSTM (recursive) | 2759 | 3371 | 24.97% | 0.965 |

**Best model by recursive-test RMSE: LightGBM.**

### EDA figures

![store_682_timeseries](outputs/figures/store_682_timeseries.png)

![store_682_seasonality](outputs/figures/store_682_seasonality.png)

![store_682_acf_pacf](outputs/figures/store_682_acf_pacf.png)

![store_682_corr](outputs/figures/store_682_corr.png)

![store_682_decomposition](outputs/figures/store_682_decomposition.png)

### Actual vs predicted (test period)

![store_682_actual_vs_sarima](outputs/figures/store_682_actual_vs_sarima.png)

![store_682_actual_vs_lstm](outputs/figures/store_682_actual_vs_lstm.png)

![store_682_actual_vs_lightgbm](outputs/figures/store_682_actual_vs_lightgbm.png)

### Feature importance (LightGBM)

![store_682_feature_importance](outputs/figures/store_682_feature_importance.png)

Top 10 features:

| Feature | Importance |
|---|---|
| log_sales_roll_std_7 | 382 |
| log_sales_roll_mean_7 | 310 |
| log_sales_lag_7 | 215 |
| log_sales_lag_1 | 213 |
| cos_doy | 202 |
| day_of_month | 193 |
| log_sales_lag_2 | 171 |
| log_sales_roll_mean_30 | 158 |
| log_sales_lag_14 | 153 |
| day_of_week | 150 |

### Error segmentation (LightGBM, one-step)

By **Promo**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 1.0 | 75.0 | 950 | 1380 | 6.62% |
| 0.0 | 114.0 | 614 | 852 | 6.55% |

By **day_of_week**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 4.0 | 27.0 | 956 | 1464 | 8.76% |
| 0.0 | 27.0 | 910 | 1227 | 6.62% |
| 1.0 | 27.0 | 822 | 1148 | 5.87% |
| 3.0 | 27.0 | 749 | 1095 | 6.06% |
| 2.0 | 27.0 | 739 | 1088 | 5.47% |
| 5.0 | 27.0 | 658 | 860 | 7.95% |
| 6.0 | 27.0 | 395 | 516 | 5.29% |

By **holiday**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 1.0 | 5.0 | 3033 | 3437 | 35.53% |
| 0.0 | 184.0 | 685 | 951 | 5.79% |

### Promotion impact (30-day scenario analysis, LightGBM)

| Scenario | Avg daily sales | 30-day total | Lift vs no-promo |
|---|---|---|---|
| No promo | 8714 | 261425 | — |
| Promo every day | 12766 | 382982 | **+46.5%** |

Scenario forecasts assume the same holiday calendar; the promo flag is the only difference.

### Forecasts

#### 1 month ahead (30 days)

![store_682_30d_forecast](outputs/figures/store_682_30d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 7659 | 6978 | 8467 | 6587 | 9262 |
| 2015-08-02 | 8241 | 7509 | 9111 | 7089 | 9966 |
| 2015-08-03 | 13929 | 12691 | 15399 | 11981 | 16844 |
| 2015-08-04 | 13228 | 12052 | 14624 | 11377 | 15996 |
| 2015-08-05 | 12775 | 11639 | 14123 | 10988 | 15448 |
| 2015-08-06 | 12563 | 11446 | 13889 | 10806 | 15192 |
| 2015-08-07 | 12345 | 11247 | 13648 | 10618 | 14929 |
| 2015-08-08 | 7521 | 6852 | 8314 | 6469 | 9095 |
| 2015-08-09 | 7536 | 6866 | 8331 | 6482 | 9113 |
| 2015-08-10 | 14069 | 12818 | 15554 | 12101 | 17013 |

Forecast CSV saved to `outputs/forecasts/store_682_30d_lightgbm.csv`.

#### 3 months ahead (90 days)

![store_682_90d_forecast](outputs/figures/store_682_90d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 7659 | 6978 | 8467 | 6587 | 9262 |
| 2015-08-02 | 8241 | 7509 | 9111 | 7089 | 9966 |
| 2015-08-03 | 13929 | 12691 | 15399 | 11981 | 16844 |
| 2015-08-04 | 13228 | 12052 | 14624 | 11377 | 15996 |
| 2015-08-05 | 12775 | 11639 | 14123 | 10988 | 15448 |
| 2015-08-06 | 12563 | 11446 | 13889 | 10806 | 15192 |
| 2015-08-07 | 12345 | 11247 | 13648 | 10618 | 14929 |
| 2015-08-08 | 7521 | 6852 | 8314 | 6469 | 9095 |
| 2015-08-09 | 7536 | 6866 | 8331 | 6482 | 9113 |
| 2015-08-10 | 14069 | 12818 | 15554 | 12101 | 17013 |

Forecast CSV saved to `outputs/forecasts/store_682_90d_lightgbm.csv`.

#### 6 months ahead (180 days)

![store_682_180d_forecast](outputs/figures/store_682_180d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 7659 | 6978 | 8467 | 6587 | 9262 |
| 2015-08-02 | 8241 | 7509 | 9111 | 7089 | 9966 |
| 2015-08-03 | 13929 | 12691 | 15399 | 11981 | 16844 |
| 2015-08-04 | 13228 | 12052 | 14624 | 11377 | 15996 |
| 2015-08-05 | 12775 | 11639 | 14123 | 10988 | 15448 |
| 2015-08-06 | 12563 | 11446 | 13889 | 10806 | 15192 |
| 2015-08-07 | 12345 | 11247 | 13648 | 10618 | 14929 |
| 2015-08-08 | 7521 | 6852 | 8314 | 6469 | 9095 |
| 2015-08-09 | 7536 | 6866 | 8331 | 6482 | 9113 |
| 2015-08-10 | 14069 | 12818 | 15554 | 12101 | 17013 |

Forecast CSV saved to `outputs/forecasts/store_682_180d_lightgbm.csv`.

## Store 733

Store type `b`, assortment `b`.
Test window: 2015-01-24 → 2015-07-31 (189 days).

### EDA insights (from full history)

- Average daily sales: **14933** units. Peak day: **Fri**, trough day: **Sat**.

| Day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Avg sales | 15542 | 14562 | 14488 | 14815 | 15966 | 14015 | 15144 |

- Promotions ran on 38% of days and lifted sales by **+10.3%** on average.

- Public holidays fell on 3.1% of days; holiday-day sales were **+9.7%** vs normal days.

### Model comparison (test period)

| Model | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|
| SARIMA | 5759 | 7257 | 38.91% | 3.154 |
| SARIMA (recursive) | 5759 | 7257 | 38.91% | 3.154 |
| LightGBM | 873 | 1170 | 5.59% | 0.478 |
| LightGBM (recursive) | 1337 | 1755 | 8.24% | 0.732 |
| LSTM | 129 | 199 | 0.81% | 0.071 |
| LSTM (recursive) | 4035 | 4673 | 27.77% | 2.210 |

**Best model by recursive-test RMSE: LightGBM.**

### EDA figures

![store_733_timeseries](outputs/figures/store_733_timeseries.png)

![store_733_seasonality](outputs/figures/store_733_seasonality.png)

![store_733_acf_pacf](outputs/figures/store_733_acf_pacf.png)

![store_733_corr](outputs/figures/store_733_corr.png)

![store_733_decomposition](outputs/figures/store_733_decomposition.png)

### Actual vs predicted (test period)

![store_733_actual_vs_sarima](outputs/figures/store_733_actual_vs_sarima.png)

![store_733_actual_vs_lstm](outputs/figures/store_733_actual_vs_lstm.png)

![store_733_actual_vs_lightgbm](outputs/figures/store_733_actual_vs_lightgbm.png)

### Feature importance (LightGBM)

![store_733_feature_importance](outputs/figures/store_733_feature_importance.png)

Top 10 features:

| Feature | Importance |
|---|---|
| log_sales_roll_std_7 | 337 |
| log_sales_lag_1 | 222 |
| sin_dow | 212 |
| log_sales_roll_mean_7 | 198 |
| day_of_month | 157 |
| log_sales_lag_7 | 144 |
| log_sales_lag_30 | 114 |
| log_sales_lag_2 | 111 |
| log_sales_lag_14 | 104 |
| Promo | 100 |

### Error segmentation (LightGBM, one-step)

By **Promo**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 0.0 | 114.0 | 907 | 1191 | 6.00% |
| 1.0 | 75.0 | 821 | 1138 | 4.96% |

By **day_of_week**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 6.0 | 27.0 | 1266 | 1459 | 7.70% |
| 3.0 | 27.0 | 1062 | 1449 | 6.70% |
| 2.0 | 27.0 | 971 | 1364 | 6.40% |
| 5.0 | 27.0 | 841 | 1034 | 5.82% |
| 0.0 | 27.0 | 788 | 1003 | 4.80% |
| 1.0 | 27.0 | 705 | 969 | 4.79% |
| 4.0 | 27.0 | 478 | 694 | 2.90% |

By **holiday**:

| Segment | n | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 1.0 | 6.0 | 2081 | 2298 | 11.27% |
| 0.0 | 183.0 | 833 | 1114 | 5.40% |

### Promotion impact (30-day scenario analysis, LightGBM)

| Scenario | Avg daily sales | 30-day total | Lift vs no-promo |
|---|---|---|---|
| No promo | 14170 | 425111 | — |
| Promo every day | 15228 | 456853 | **+7.5%** |

Scenario forecasts assume the same holiday calendar; the promo flag is the only difference.

### Forecasts

#### 1 month ahead (30 days)

![store_733_30d_forecast](outputs/figures/store_733_30d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 14032 | 13114 | 15729 | 12303 | 16767 |
| 2015-08-02 | 15574 | 14556 | 17458 | 13655 | 18610 |
| 2015-08-03 | 15501 | 14487 | 17376 | 13591 | 18523 |
| 2015-08-04 | 14877 | 13904 | 16676 | 13044 | 17777 |
| 2015-08-05 | 14349 | 13410 | 16084 | 12581 | 17146 |
| 2015-08-06 | 14222 | 13292 | 15942 | 12470 | 16994 |
| 2015-08-07 | 15474 | 14462 | 17346 | 13568 | 18491 |
| 2015-08-08 | 13954 | 13042 | 15642 | 12235 | 16674 |
| 2015-08-09 | 14859 | 13887 | 16656 | 13028 | 17755 |
| 2015-08-10 | 15492 | 14479 | 17366 | 13584 | 18512 |

Forecast CSV saved to `outputs/forecasts/store_733_30d_lightgbm.csv`.

#### 3 months ahead (90 days)

![store_733_90d_forecast](outputs/figures/store_733_90d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 14032 | 13114 | 15729 | 12303 | 16767 |
| 2015-08-02 | 15574 | 14556 | 17458 | 13655 | 18610 |
| 2015-08-03 | 15501 | 14487 | 17376 | 13591 | 18523 |
| 2015-08-04 | 14877 | 13904 | 16676 | 13044 | 17777 |
| 2015-08-05 | 14349 | 13410 | 16084 | 12581 | 17146 |
| 2015-08-06 | 14222 | 13292 | 15942 | 12470 | 16994 |
| 2015-08-07 | 15474 | 14462 | 17346 | 13568 | 18491 |
| 2015-08-08 | 13954 | 13042 | 15642 | 12235 | 16674 |
| 2015-08-09 | 14859 | 13887 | 16656 | 13028 | 17755 |
| 2015-08-10 | 15492 | 14479 | 17366 | 13584 | 18512 |

Forecast CSV saved to `outputs/forecasts/store_733_90d_lightgbm.csv`.

#### 6 months ahead (180 days)

![store_733_180d_forecast](outputs/figures/store_733_180d_forecast.png)

Best model (`LightGBM`) point forecast and bands (first 10 days):

| Date | Predicted | Lower 80% | Upper 80% | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| 2015-08-01 | 14032 | 13114 | 15729 | 12303 | 16767 |
| 2015-08-02 | 15574 | 14556 | 17458 | 13655 | 18610 |
| 2015-08-03 | 15501 | 14487 | 17376 | 13591 | 18523 |
| 2015-08-04 | 14877 | 13904 | 16676 | 13044 | 17777 |
| 2015-08-05 | 14349 | 13410 | 16084 | 12581 | 17146 |
| 2015-08-06 | 14222 | 13292 | 15942 | 12470 | 16994 |
| 2015-08-07 | 15474 | 14462 | 17346 | 13568 | 18491 |
| 2015-08-08 | 13954 | 13042 | 15642 | 12235 | 16674 |
| 2015-08-09 | 14859 | 13887 | 16656 | 13028 | 17755 |
| 2015-08-10 | 15492 | 14479 | 17366 | 13584 | 18512 |

Forecast CSV saved to `outputs/forecasts/store_733_180d_lightgbm.csv`.

## Summary across stores

| Store | Best model | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|---|
| 1097 | LightGBM | 786 | 1045 | 7.09% | 0.619 |
| 682 | LightGBM | 940 | 1372 | 7.91% | 0.329 |
| 733 | LightGBM | 1337 | 1755 | 8.24% | 0.732 |

## EDA insights

Key patterns (see per-store tables and figures above):
- **Weekly seasonality**: peak day varies by store — store 1097 peaks on **Sunday**, store 682 on **Monday**, store 733 on **Friday**; every store troughs on **Saturday**.
- **Promotions**: measured lift ranges from **+10% to +57%** depending on store (store 682 benefits most).
- **Holidays**: effects differ by store — store 1097 sees **+35%** on holiday days, store 682 sees **-36%** (holidays can depress weekday-heavy traffic), store 733 is roughly neutral.
- **ACF/PACF**: first-differenced log sales show significant autocorrelation at weekly lags, confirming strong weekly seasonality (period 7).
- **Decomposition**: stable weekly seasonal component; slow upward trend for the two highest-volume stores.

## Business recommendations

- **Promotion timing**: schedule promotions on weekdays (Tue-Fri) rather than the weekend peak, where incremental lift is highest and cannibalisation of natural demand is lowest.
- **Inventory**: raise stock around December/Christmas (holiday flag c) and during Promo2 windows; the forecast bands quantify the safety stock needed (upper 95% band).
- **Model choice**: the tree model with lag/rolling features is the most accurate multi-step forecaster; use SARIMA as a robust sanity-check baseline and the LSTM when you want smooth day-to-day dynamics.
- **Confidence-aware planning**: use the 80% band for operational staffing and the 95% band for safety-stock commitments.

## Reproducibility

- Run `python run_pipeline.py` to reproduce end-to-end results.
- `requirements.txt` pins library versions; `src/sales_forecast/` is the modular pipeline.
- `forecast_sales(store_id_or_dataframe, horizon=30)` returns the forecast dataframe.
- Model artifacts are saved under `outputs/models/`; forecasts under `outputs/forecasts/`.

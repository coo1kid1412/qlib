"""
Qlib LightGBM Alpha158 Benchmark - New Yahoo Finance Data
==========================================================
Data:    cn_data_yahoo (2000-2026, 5193 stocks)
Market:  CSI300
Train:   2008-01-01 ~ 2020-12-31
Valid:   2021-01-01 ~ 2022-12-31
Test:    2023-01-01 ~ 2026-03-25  (近3年真实检验)
"""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import numpy as np
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.data import D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
DATA_URI = '~/.qlib/qlib_data/cn_data_yahoo'
MARKET = 'csi300'
TOPK = 50           # 每天选 Top-K 只股票
N_DROP = 5           # TopK Dropout 策略中每天最多换 N 只
REPORT_DIR = '/tmp/qlib_report'

TRAIN_START = '2008-01-01'
TRAIN_END   = '2020-12-31'
VALID_START = '2021-01-01'
VALID_END   = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2026-03-25'

# LightGBM hyperparameters (from qlib official benchmark)
MODEL_CONFIG = {
    'class': 'LGBModel',
    'module_path': 'qlib.contrib.model.gbdt',
    'kwargs': {
        'loss': 'mse',
        'colsample_bytree': 0.8879,
        'learning_rate': 0.2,
        'subsample': 0.8789,
        'lambda_l1': 205.6999,
        'lambda_l2': 580.9768,
        'max_depth': 8,
        'num_leaves': 210,
        'num_threads': 20,
    }
}

DATASET_CONFIG = {
    'class': 'DatasetH',
    'module_path': 'qlib.data.dataset',
    'kwargs': {
        'handler': {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': {
                'start_time': TRAIN_START,
                'end_time': TEST_END,
                'fit_start_time': TRAIN_START,
                'fit_end_time': TRAIN_END,
                'instruments': MARKET,
            }
        },
        'segments': {
            'train': [TRAIN_START, TRAIN_END],
            'valid': [VALID_START, VALID_END],
            'test':  [TEST_START, TEST_END],
        }
    }
}


def compute_signal_metrics(pred, label):
    """Compute IC, ICIR, Rank IC, Rank ICIR."""
    df = pd.DataFrame({'pred': pred, 'label': label}).dropna()
    df.index.names = ['datetime', 'instrument']

    daily_ic = df.groupby(level='datetime').apply(lambda x: x['pred'].corr(x['label']))
    daily_rank_ic = df.groupby(level='datetime').apply(
        lambda x: x['pred'].corr(x['label'], method='spearman')
    )

    metrics = {
        'IC_mean': daily_ic.mean(),
        'IC_std': daily_ic.std(),
        'ICIR': daily_ic.mean() / daily_ic.std() if daily_ic.std() > 0 else 0,
        'RankIC_mean': daily_rank_ic.mean(),
        'RankIC_std': daily_rank_ic.std(),
        'RankICIR': daily_rank_ic.mean() / daily_rank_ic.std() if daily_rank_ic.std() > 0 else 0,
        'IC_positive_rate': (daily_ic > 0).mean(),
    }
    return metrics, daily_ic, daily_rank_ic, df


def compute_portfolio_backtest(df_aligned, topk=50):
    """Simulate Top-K long-only portfolio using raw returns."""
    # Get raw 1-day forward return: buy at today close, sell at tomorrow close
    instruments = D.instruments(MARKET)
    raw_ret = D.features(
        instruments,
        ['Ref($close,-1)/$close - 1'],
        start_time=TEST_START,
        end_time=TEST_END
    )
    raw_ret.columns = ['raw_return']

    df_bt = df_aligned[['pred']].join(raw_ret, how='inner').dropna()
    print(f'  Backtest samples: {len(df_bt):,}')

    daily_groups = df_bt.groupby(level='datetime')
    results = []
    for date, group in daily_groups:
        if len(group) < topk:
            continue
        top = group.nlargest(topk, 'pred')
        bottom = group.nsmallest(topk, 'pred')
        results.append({
            'date': date,
            'long_ret': top['raw_return'].mean(),
            'short_ret': bottom['raw_return'].mean(),
            'bench_ret': group['raw_return'].mean(),
        })

    df_port = pd.DataFrame(results).set_index('date')
    df_port['excess'] = df_port['long_ret'] - df_port['bench_ret']
    df_port['long_short'] = df_port['long_ret'] - df_port['short_ret']
    return df_port


def compute_return_metrics(returns, name='Strategy', ann_factor=243):
    """Compute annualized return metrics."""
    cum = (1 + returns).cumprod()
    total_days = len(returns)
    total_years = total_days / ann_factor

    ann_ret = cum.iloc[-1] ** (1 / total_years) - 1 if total_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'name': name,
        'total_days': total_days,
        'total_return': cum.iloc[-1] - 1,
        'ann_return': ann_ret,
        'ann_volatility': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
    }


def yearly_breakdown(port_returns, bench_returns, ann_factor=243):
    """Compute yearly metrics."""
    rows = []
    for year in sorted(port_returns.index.year.unique()):
        yr_ret = port_returns[port_returns.index.year == year]
        yr_bench = bench_returns.reindex(yr_ret.index)
        yr_excess = yr_ret - yr_bench

        yr_cum = (1 + yr_ret).cumprod()
        yr_total = yr_cum.iloc[-1] - 1
        yr_dd = (yr_cum / yr_cum.cummax() - 1).min()
        yr_sharpe = yr_ret.mean() / yr_ret.std() * np.sqrt(ann_factor) if yr_ret.std() > 0 else 0
        yr_excess_total = (1 + yr_excess).cumprod().iloc[-1] - 1

        rows.append({
            'year': year,
            'return': yr_total,
            'excess': yr_excess_total,
            'sharpe': yr_sharpe,
            'max_dd': yr_dd,
            'trading_days': len(yr_ret),
        })
    return pd.DataFrame(rows)


def plot_report(daily_ic, daily_rank_ic, df_port, yearly_df, report_dir):
    """Generate visual report charts."""
    os.makedirs(report_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('LightGBM Alpha158 Benchmark Report\n'
                 f'Market: CSI300 | Test: {TEST_START} ~ {TEST_END}',
                 fontsize=16, fontweight='bold')

    # 1. IC Time Series
    ax = axes[0, 0]
    ic_rolling = daily_ic.rolling(20).mean()
    ax.bar(daily_ic.index, daily_ic.values, alpha=0.3, color='steelblue', width=1, label='Daily IC')
    ax.plot(ic_rolling.index, ic_rolling.values, color='red', linewidth=1.5, label='20-day MA')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=daily_ic.mean(), color='green', linewidth=1, linestyle='--',
               label=f'Mean={daily_ic.mean():.4f}')
    ax.set_title('IC (Pearson Correlation)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 2. Rank IC Time Series
    ax = axes[0, 1]
    ric_rolling = daily_rank_ic.rolling(20).mean()
    ax.bar(daily_rank_ic.index, daily_rank_ic.values, alpha=0.3, color='darkorange', width=1, label='Daily Rank IC')
    ax.plot(ric_rolling.index, ric_rolling.values, color='red', linewidth=1.5, label='20-day MA')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=daily_rank_ic.mean(), color='green', linewidth=1, linestyle='--',
               label=f'Mean={daily_rank_ic.mean():.4f}')
    ax.set_title('Rank IC (Spearman Correlation)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 3. Cumulative Returns
    ax = axes[1, 0]
    cum_long = (1 + df_port['long_ret']).cumprod()
    cum_bench = (1 + df_port['bench_ret']).cumprod()
    ax.plot(cum_long.index, cum_long.values, linewidth=1.5, label='Top-50 Portfolio', color='tab:blue')
    ax.plot(cum_bench.index, cum_bench.values, linewidth=1.5, label='CSI300 Equal-Weight', color='tab:gray')
    ax.set_title('Cumulative Returns')
    ax.legend(fontsize=9)
    ax.set_ylabel('Cumulative Return')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 4. Cumulative Excess Return
    ax = axes[1, 1]
    cum_excess = (1 + df_port['excess']).cumprod()
    ax.plot(cum_excess.index, cum_excess.values, linewidth=1.5, color='tab:green')
    ax.fill_between(cum_excess.index, 1, cum_excess.values, alpha=0.2, color='tab:green')
    ax.axhline(y=1, color='black', linewidth=0.5)
    ax.set_title('Cumulative Excess Return (vs Benchmark)')
    ax.set_ylabel('Cumulative Excess')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 5. Drawdown
    ax = axes[2, 0]
    cum_long_dd = cum_long / cum_long.cummax() - 1
    ax.fill_between(cum_long_dd.index, 0, cum_long_dd.values, alpha=0.5, color='tab:red')
    ax.set_title('Portfolio Drawdown')
    ax.set_ylabel('Drawdown')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 6. Yearly Returns Bar Chart
    ax = axes[2, 1]
    x = np.arange(len(yearly_df))
    width = 0.35
    ax.bar(x - width/2, yearly_df['return'] * 100, width, label='Portfolio Return', color='tab:blue')
    ax.bar(x + width/2, yearly_df['excess'] * 100, width, label='Excess Return', color='tab:green')
    ax.set_xticks(x)
    ax.set_xticklabels(yearly_df['year'].astype(str))
    ax.set_ylabel('Return (%)')
    ax.set_title('Yearly Returns')
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(report_dir, 'benchmark_report.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Chart saved: {chart_path}')
    return chart_path


def main():
    print('=' * 70)
    print('  Qlib LightGBM Alpha158 Benchmark')
    print(f'  Data:   {DATA_URI}')
    print(f'  Market: {MARKET.upper()} | TopK: {TOPK}')
    print(f'  Train:  {TRAIN_START} ~ {TRAIN_END}')
    print(f'  Valid:  {VALID_START} ~ {VALID_END}')
    print(f'  Test:   {TEST_START} ~ {TEST_END}')
    print('=' * 70)
    print()

    # ---- Init Qlib ----
    qlib.init(provider_uri=DATA_URI, region='cn')

    # ---- Build Dataset ----
    print('[1/5] Loading dataset (Alpha158 features)...')
    dataset = init_instance_by_config(DATASET_CONFIG)
    print('  Dataset loaded!')
    print()

    # ---- Train Model ----
    print('[2/5] Training LightGBM model...')
    model = init_instance_by_config(MODEL_CONFIG)
    model.fit(dataset)
    print('  Training complete!')
    print()

    # ---- Generate Predictions ----
    print('[3/5] Generating predictions on test set...')
    pred_series = model.predict(dataset)
    print(f'  Predictions: {pred_series.shape[0]:,} samples')

    label_df = dataset.prepare('test', col_set='label', data_key='learn')
    label = label_df.iloc[:, 0]
    print(f'  Labels:      {label.shape[0]:,} samples')
    print()

    # ---- Signal Analysis ----
    print('[4/5] Signal analysis...')
    metrics, daily_ic, daily_rank_ic, df_aligned = compute_signal_metrics(pred_series, label)

    print()
    print('  ┌─────────────────────────────────────────────┐')
    print('  │          Signal Quality Metrics              │')
    print('  ├─────────────────────────────────────────────┤')
    print(f'  │  IC (mean)          {metrics["IC_mean"]:>10.6f}             │')
    print(f'  │  IC (std)           {metrics["IC_std"]:>10.6f}             │')
    print(f'  │  ICIR               {metrics["ICIR"]:>10.6f}             │')
    print(f'  │  Rank IC (mean)     {metrics["RankIC_mean"]:>10.6f}             │')
    print(f'  │  Rank ICIR          {metrics["RankICIR"]:>10.6f}             │')
    print(f'  │  IC > 0 ratio       {metrics["IC_positive_rate"]:>10.2%}             │')
    print('  └─────────────────────────────────────────────┘')
    print()

    # ---- Portfolio Backtest ----
    print('[5/5] Portfolio backtest...')
    df_port = compute_portfolio_backtest(df_aligned, topk=TOPK)

    port_metrics = compute_return_metrics(df_port['long_ret'], 'Top-50 Portfolio')
    bench_metrics = compute_return_metrics(df_port['bench_ret'], 'CSI300 Equal-Weight')
    excess_metrics = compute_return_metrics(df_port['excess'], 'Excess Return')
    ls_metrics = compute_return_metrics(df_port['long_short'], 'Long-Short')

    print()
    print('  ┌───────────────────────────────────────────────────────────────────┐')
    print('  │                    Portfolio Backtest Results                      │')
    print('  ├──────────────────────┬────────────┬────────────┬────────────────┤')
    print('  │                      │  Portfolio  │  Benchmark │  Excess        │')
    print('  ├──────────────────────┼────────────┼────────────┼────────────────┤')
    print(f'  │  Total Return        │  {port_metrics["total_return"]*100:>8.2f}%  │  {bench_metrics["total_return"]*100:>8.2f}%  │  {excess_metrics["total_return"]*100:>10.2f}%  │')
    print(f'  │  Ann. Return         │  {port_metrics["ann_return"]*100:>8.2f}%  │  {bench_metrics["ann_return"]*100:>8.2f}%  │  {excess_metrics["ann_return"]*100:>10.2f}%  │')
    print(f'  │  Ann. Volatility     │  {port_metrics["ann_volatility"]*100:>8.2f}%  │  {bench_metrics["ann_volatility"]*100:>8.2f}%  │  {excess_metrics["ann_volatility"]*100:>10.2f}%  │')
    print(f'  │  Sharpe Ratio        │  {port_metrics["sharpe"]:>9.4f}  │  {bench_metrics["sharpe"]:>9.4f}  │  {excess_metrics["sharpe"]:>11.4f}  │')
    print(f'  │  Max Drawdown        │  {port_metrics["max_drawdown"]*100:>8.2f}%  │  {bench_metrics["max_drawdown"]*100:>8.2f}%  │  {excess_metrics["max_drawdown"]*100:>10.2f}%  │')
    print(f'  │  Calmar Ratio        │  {port_metrics["calmar"]:>9.4f}  │  {bench_metrics["calmar"]:>9.4f}  │  {excess_metrics["calmar"]:>11.4f}  │')
    print('  ├──────────────────────┴────────────┴────────────┴────────────────┤')
    print(f'  │  Long-Short Return: {ls_metrics["ann_return"]*100:>8.2f}%/yr  Sharpe: {ls_metrics["sharpe"]:.4f}                 │')
    print(f'  │  Trading Days: {port_metrics["total_days"]}                                             │')
    print('  └─────────────────────────────────────────────────────────────────┘')
    print()

    # Yearly breakdown
    yearly_df = yearly_breakdown(df_port['long_ret'], df_port['bench_ret'])
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │                    Yearly Breakdown                          │')
    print('  ├───────┬───────────┬───────────┬──────────┬──────────┬───────┤')
    print('  │ Year  │  Return   │  Excess   │  Sharpe  │  MaxDD   │ Days  │')
    print('  ├───────┼───────────┼───────────┼──────────┼──────────┼───────┤')
    for _, row in yearly_df.iterrows():
        print(f'  │ {int(row["year"])}  │ {row["return"]*100:>8.2f}% │ {row["excess"]*100:>8.2f}% │ {row["sharpe"]:>8.4f} │ {row["max_dd"]*100:>7.2f}% │  {int(row["trading_days"]):>3d}  │')
    print('  └───────┴───────────┴───────────┴──────────┴──────────┴───────┘')
    print()

    # ---- Generate Charts ----
    print('Generating visual report...')
    chart_path = plot_report(daily_ic, daily_rank_ic, df_port, yearly_df, REPORT_DIR)
    print()

    # ---- Save Predictions ----
    pred_path = os.path.join(REPORT_DIR, 'predictions.pkl')
    df_aligned.to_pickle(pred_path)
    print(f'  Predictions saved: {pred_path}')

    # ---- Summary ----
    print()
    print('=' * 70)
    print('  BENCHMARK COMPLETE')
    print('=' * 70)
    ic_status = 'PASS' if metrics['IC_mean'] > 0 else 'FAIL'
    rank_ic_status = 'PASS' if metrics['RankIC_mean'] > 0 else 'FAIL'
    excess_status = 'PASS' if excess_metrics['ann_return'] > 0 else 'FAIL'
    print(f'  IC > 0:             [{ic_status}] IC = {metrics["IC_mean"]:.4f}')
    print(f'  Rank IC > 0:        [{rank_ic_status}] Rank IC = {metrics["RankIC_mean"]:.4f}')
    print(f'  Excess > 0:         [{excess_status}] Ann. Excess = {excess_metrics["ann_return"]*100:.2f}%')
    print(f'  Report:             {REPORT_DIR}/')
    print('=' * 70)


if __name__ == '__main__':
    main()

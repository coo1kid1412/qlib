"""
Download A-share daily data using baostock and save as qlib-compatible CSV.
baostock uses its own protocol (not HTTP), so it bypasses proxy/firewall issues.

Usage:
    python baostock_to_qlib.py --output_dir ~/.qlib/stock_data/source/cn_baostock \
                               --start_date 2000-01-01 --end_date 2026-03-26
"""
import argparse
import sys
from pathlib import Path

import baostock as bs
import pandas as pd
from tqdm import tqdm


def get_all_stocks(date=None):
    """Get all A-share stock codes for a given date."""
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    rs = bs.query_all_stock(day=date)
    stocks = []
    while (rs.error_code == "0") and rs.next():
        row = rs.get_row_data()
        code = row[0]  # e.g. sh.600000
        # Only keep stocks (exclude indices and funds)
        if code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3") or code.startswith("sh.68") or code.startswith("sz.00"):
            stocks.append(code)
    return sorted(set(stocks))


def download_single_stock(code, start_date, end_date, output_dir):
    """Download daily data for a single stock and save as CSV."""
    # Convert baostock code (sh.600000) to qlib symbol (sh600000)
    symbol = code.replace(".", "")
    output_path = Path(output_dir) / f"{symbol}.csv"

    if output_path.exists() and output_path.stat().st_size > 100:
        return symbol, "skipped"

    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",  # 前复权
        )
        if rs.error_code != "0":
            return symbol, f"error:{rs.error_msg}"

        data_list = []
        while (rs.error_code == "0") and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return symbol, "empty"

        df = pd.DataFrame(data_list, columns=rs.fields)

        # Convert types
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add symbol column
        df["symbol"] = symbol

        # Keep needed columns
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "symbol"]
        df = df[[c for c in cols if c in df.columns]]

        # Remove rows with empty/zero data
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        if df.empty:
            return symbol, "empty_after_clean"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return symbol, f"ok:{len(df)}"

    except Exception as e:
        return symbol, f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description="Download A-share data via baostock")
    parser.add_argument("--output_dir", type=str, default="~/.qlib/stock_data/source/cn_baostock")
    parser.add_argument("--start_date", type=str, default="2000-01-01")
    parser.add_argument("--end_date", type=str, default="2026-03-26")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Login
    lg = bs.login()
    if lg.error_code != "0":
        print(f"Login failed: {lg.error_msg}")
        sys.exit(1)
    print("baostock login success")

    # Get stock list
    print("Fetching stock list...")
    stocks = get_all_stocks()
    print(f"Total stocks: {len(stocks)}")

    ok_count = 0
    skip_count = 0
    err_count = 0
    empty_count = 0

    for code in tqdm(stocks, desc="Downloading"):
        symbol, status = download_single_stock(code, args.start_date, args.end_date, output_dir)
        if status.startswith("ok"):
            ok_count += 1
        elif status == "skipped":
            skip_count += 1
        elif "empty" in status:
            empty_count += 1
        else:
            err_count += 1
            if err_count <= 20:
                tqdm.write(f"  {symbol}: {status}")

    bs.logout()
    print(f"\nDone! ok={ok_count}, skipped={skip_count}, empty={empty_count}, error={err_count}")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

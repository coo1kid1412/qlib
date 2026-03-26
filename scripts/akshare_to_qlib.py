"""
Download A-share daily data using akshare and convert to qlib-compatible CSV format.
Usage:
    python akshare_to_qlib.py --output_dir ~/.qlib/stock_data/source/cn_akshare \
                              --start_date 20000101 --end_date 20260326 \
                              --max_workers 1 --delay 0.3
"""
import argparse
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd
from tqdm import tqdm


def get_all_stock_codes():
    """Get all A-share stock codes from akshare."""
    df = ak.stock_info_a_code_name()
    codes = df["code"].tolist()
    return codes


def download_single_stock(code, start_date, end_date, output_dir, delay=0.3):
    """Download daily data for a single stock and save as CSV."""
    # Map code to qlib symbol format: sh600000 or sz000001
    if code.startswith("6") or code.startswith("9"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"

    output_path = Path(output_dir) / f"{symbol}.csv"
    if output_path.exists() and output_path.stat().st_size > 100:
        return symbol, "skipped"

    try:
        time.sleep(delay)
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date,
            adjust="qfq"  # 前复权
        )
        if df is None or df.empty:
            return symbol, "empty"

        # Rename columns to qlib-compatible format
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        })

        # Keep only needed columns
        cols = ["date", "open", "close", "high", "low", "volume", "amount"]
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]

        # Add symbol column
        df["symbol"] = symbol

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return symbol, f"ok:{len(df)}"

    except Exception as e:
        return symbol, f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description="Download A-share data via akshare")
    parser.add_argument("--output_dir", type=str, default="~/.qlib/stock_data/source/cn_akshare")
    parser.add_argument("--start_date", type=str, default="20000101")
    parser.add_argument("--end_date", type=str, default="20260326")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching stock list...")
    codes = get_all_stock_codes()
    print(f"Total stocks: {len(codes)}")

    ok_count = 0
    skip_count = 0
    err_count = 0
    empty_count = 0

    if args.max_workers <= 1:
        # Sequential download
        for code in tqdm(codes, desc="Downloading"):
            symbol, status = download_single_stock(
                code, args.start_date, args.end_date, output_dir, args.delay
            )
            if status.startswith("ok"):
                ok_count += 1
            elif status == "skipped":
                skip_count += 1
            elif status == "empty":
                empty_count += 1
            else:
                err_count += 1
                if err_count <= 10:
                    print(f"\n  {symbol}: {status}")
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    download_single_stock, code, args.start_date, args.end_date, output_dir, args.delay
                ): code for code in codes
            }
            for future in tqdm(as_completed(futures), total=len(codes), desc="Downloading"):
                symbol, status = future.result()
                if status.startswith("ok"):
                    ok_count += 1
                elif status == "skipped":
                    skip_count += 1
                elif status == "empty":
                    empty_count += 1
                else:
                    err_count += 1
                    if err_count <= 10:
                        print(f"\n  {symbol}: {status}")

    print(f"\nDone! ok={ok_count}, skipped={skip_count}, empty={empty_count}, error={err_count}")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

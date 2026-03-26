"""
Create index constituent files (csi300.txt, csi500.txt) for qlib using baostock.
Queries historical constituent data at each rebalance date to build proper date ranges.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import baostock as bs
from datetime import datetime, timedelta
from collections import defaultdict

QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data_yahoo/instruments")

def normalize_symbol(baostock_code):
    """Convert baostock code (sh.600519) to qlib format (SH600519)"""
    parts = baostock_code.split('.')
    return f"{parts[0].upper()}{parts[1]}"

def get_rebalance_dates():
    """
    CSI300/CSI500 rebalance twice a year:
    - June (second Friday of June, effective next trading day)
    - December (second Friday of December, effective next trading day)
    We sample monthly to catch changes more granularly.
    """
    dates = []
    # Sample every month from 2005-01 to 2026-03
    for year in range(2005, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 3:
                break
            # Use 15th of each month as sample date
            dates.append(f"{year}-{month:02d}-15")
    return dates

def query_index_constituents(query_func, index_name):
    """Query index constituents across multiple dates to build history."""
    dates = get_rebalance_dates()
    
    # Track when each stock enters/exits the index
    # stock_periods: {symbol: [(start_date, end_date), ...]}
    stock_in_index = {}  # {symbol: current_start_date}
    stock_periods = defaultdict(list)
    prev_members = set()
    last_valid_date = None
    
    print(f"\nQuerying {index_name} constituents across {len(dates)} sample dates...")
    
    for i, date in enumerate(dates):
        if i % 24 == 0:
            print(f"  Processing {date}...")
        
        rs = query_func(date=date)
        members = set()
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1]  # code field
            symbol = normalize_symbol(code)
            members.add(symbol)
        
        if not members:
            continue
        
        last_valid_date = date
        
        # New members (entered the index)
        for sym in members - prev_members:
            stock_in_index[sym] = date
        
        # Removed members (exited the index)
        for sym in prev_members - members:
            if sym in stock_in_index:
                stock_periods[sym].append((stock_in_index[sym], date))
                del stock_in_index[sym]
        
        prev_members = members
    
    # Close all still-active memberships with last valid date or today
    end_date = "2026-03-25"
    for sym, start in stock_in_index.items():
        stock_periods[sym].append((start, end_date))
    
    return dict(stock_periods)

def write_instruments_file(stock_periods, filename):
    """Write instruments file in qlib format: SYMBOL\tSTART_DATE\tEND_DATE"""
    filepath = os.path.join(QLIB_DIR, filename)
    lines = []
    for symbol in sorted(stock_periods.keys()):
        for start, end in stock_periods[symbol]:
            lines.append(f"{symbol}\t{start}\t{end}")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    unique_stocks = len(stock_periods)
    total_entries = len(lines)
    print(f"  Written {filepath}: {unique_stocks} unique stocks, {total_entries} entries")
    return filepath

def main():
    print("=" * 60)
    print("Creating index constituent files for qlib")
    print("=" * 60)
    
    bs.login()
    
    # CSI300
    print("\n[1/2] CSI300 (沪深300)")
    csi300_periods = query_index_constituents(bs.query_hs300_stocks, "CSI300")
    if csi300_periods:
        write_instruments_file(csi300_periods, "csi300.txt")
    else:
        print("  WARNING: No CSI300 data retrieved!")
    
    # CSI500
    print("\n[2/2] CSI500 (中证500)")
    csi500_periods = query_index_constituents(bs.query_zz500_stocks, "CSI500")
    if csi500_periods:
        write_instruments_file(csi500_periods, "csi500.txt")
    else:
        print("  WARNING: No CSI500 data retrieved!")
    
    bs.logout()
    
    # Verify
    print("\n" + "=" * 60)
    print("Verification:")
    for fname in ["csi300.txt", "csi500.txt"]:
        fpath = os.path.join(QLIB_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                lines = f.readlines()
            symbols = set(l.split('\t')[0] for l in lines)
            print(f"  {fname}: {len(lines)} entries, {len(symbols)} unique stocks")
            print(f"    Sample: {lines[0].strip()}")
            print(f"    Sample: {lines[-1].strip()}")
        else:
            print(f"  {fname}: NOT FOUND")
    
    print("\nDone!")

if __name__ == '__main__':
    main()

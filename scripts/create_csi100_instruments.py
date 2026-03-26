"""
Create CSI100 (中证100) instruments file by:
1. Getting current constituents from csindex.com.cn Excel
2. Parsing HTML table (index=1) from each announcement for CSI100 add/remove
3. Building full date-range instruments file
"""
import os
import re
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
from io import BytesIO, StringIO
from collections import defaultdict

QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data_yahoo/instruments")
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
INDEX_CODE = "000903"
BENCH_START = "2006-05-29"
# CSI100 changes are in the 2nd HTML table (index=1), after CSI300 (index=0)
HTML_TABLE_INDEX = 1


def normalize_symbol(symbol):
    """Convert numeric code to qlib format (SH600519)"""
    s = str(symbol).strip()
    # Skip HK stocks
    if '.HK' in s:
        return None
    code = re.sub(r'\D', '', s)
    if not code:
        return None
    # Pad to 6 digits (Excel reads 000002 as integer 2)
    code = f"{int(code):06d}"
    if code.startswith("6") or code.startswith("688") or code.startswith("689"):
        return f"SH{code}"
    else:
        return f"SZ{code}"


def get_current_constituents():
    """Get current CSI100 constituents from csindex.com.cn"""
    url = f"https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{INDEX_CODE}cons.xls"
    print(f"  Downloading current constituents from csindex.com.cn...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    df = pd.read_excel(BytesIO(r.content))
    symbols = []
    for _, row in df.iterrows():
        sym = normalize_symbol(row.iloc[4])  # 成份券代码
        if sym:
            symbols.append(sym)

    print(f"  Current CSI100: {len(symbols)} stocks")
    return symbols


def get_announcement_list():
    """Get all historical change announcement IDs"""
    url = (
        "https://www.csindex.com.cn/csindex-home/search/search-content?"
        "lang=cn&searchInput=%E5%85%B3%E4%BA%8E%E8%B0%83%E6%95%B4%E6%B2%AA%E6%B7%B1300"
        "%E5%92%8C%E4%B8%AD%E8%AF%81%E9%A6%99%E6%B8%AF100%E7%AD%89%E6%8C%87%E6%95%B0"
        "%E6%A0%B7%E6%9C%AC&pageNum=1&pageSize=50&sortField=date&dateRange=all&contentType=announcement"
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    announcements = []
    for item in data['data']:
        announcements.append({
            'id': item['id'],
            'date': item['itemDate'],
        })

    announcements.sort(key=lambda x: x['date'])
    print(f"  Found {len(announcements)} announcements")
    return announcements


def parse_announcement(ann_id):
    """Parse a single announcement to extract CSI100 add/remove changes from HTML table"""
    url = f"https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById?id={ann_id}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    resp = r.json()['data']

    content = resp.get('content', '')

    # Extract effective date
    date_matches = re.findall(r'(\d{4})\s*年\s*(\d+)\s*月\s*(\d+)\s*日', content)
    if not date_matches:
        return None, [], []

    effective_date = f"{date_matches[0][0]}-{int(date_matches[0][1]):02d}-{int(date_matches[0][2]):02d}"

    # If "盘后"/"市后"/"收市后", effective next trading day
    if "盘后" in content or "市后" in content or "收市后" in content:
        eff_ts = pd.Timestamp(effective_date)
        next_day = eff_ts + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        effective_date = next_day.strftime('%Y-%m-%d')

    adds = []
    removes = []

    # Parse HTML tables - CSI100 is at change_table index 1
    try:
        tables = pd.read_html(StringIO(content))
    except Exception:
        return effective_date, [], []

    # Find valid change tables (4 columns with header rows containing 调出/调入)
    change_tables = []
    for tbl in tables:
        if tbl.shape[1] == 4 and tbl.shape[0] >= 3:
            first_val = str(tbl.iloc[0, 0]) if pd.notna(tbl.iloc[0, 0]) else ''
            if '调出' in first_val or '名单' in first_val:
                change_tables.append(tbl)

    if len(change_tables) <= HTML_TABLE_INDEX:
        return effective_date, [], []

    tbl = change_tables[HTML_TABLE_INDEX]

    # Extract removes (col 0) and adds (col 2), skip 2 header rows
    for _, row in tbl.iloc[2:].iterrows():
        remove_code = row.iloc[0]
        add_code = row.iloc[2]

        if pd.notna(remove_code):
            sym = normalize_symbol(remove_code)
            if sym:
                removes.append(sym)

        if pd.notna(add_code):
            sym = normalize_symbol(add_code)
            if sym:
                adds.append(sym)

    return effective_date, adds, removes


def build_instruments(current_members, changes):
    """
    Build instruments file using two strategies:
    1. Pre-2022: Forward simulation from bench_start using parsed changes
    2. Post-2022: Current constituents from 2022-06-13 revision to now
       (individual rebalance announcements for CSI100 are not publicly searchable after 2021)
    """
    REVISION_DATE = "2022-06-13"  # CSI100 methodology revision date
    END_DATE = "2026-03-25"

    intervals = defaultdict(list)

    # === Part 1: Forward simulation for pre-2022 history ===
    # Start with an initial member set, then apply changes forward
    # We need to reconstruct who was in the index at bench_start
    # Strategy: start from the last known pre-2022 state and work backwards to bench_start,
    # then forward to build intervals

    # Split changes into pre-revision and post-revision
    pre_changes = [(d, a, r) for d, a, r in changes if d < REVISION_DATE]

    if pre_changes:
        # Build initial members at last pre-change date by working backwards
        # Start from: just before revision, apply pre-changes in reverse
        # We don't know the exact members at revision, so we reconstruct from bench_start forward

        # Forward approach: start with unknown initial set, track add/remove
        # We know the first change date - everyone who was removed before being added
        # must have been an original member

        all_added = set()
        all_removed = set()
        original_members = set()

        for eff_date, adds, removes in pre_changes:
            for sym in removes:
                if sym not in all_added:
                    original_members.add(sym)
            all_added.update(adds)
            all_removed.update(removes)

        # Also, stocks that were added and never removed are NOT original members
        # Stocks in the first change's remove list were definitely original members
        # This is approximate but reasonable

        # Forward simulate
        members = set(original_members)
        prev_date = BENCH_START

        for eff_date, adds, removes in pre_changes:
            # Record interval for stocks being removed
            for sym in removes:
                if sym in members:
                    intervals[sym].append((prev_date, eff_date))
                    members.discard(sym)
            # Add new stocks
            members.update(adds)
            prev_date = eff_date

        # Close remaining pre-revision members at revision date
        for sym in members:
            intervals[sym].append((prev_date, REVISION_DATE))

    # === Part 2: Current constituents from revision date to now ===
    for sym in current_members:
        intervals[sym].append((REVISION_DATE, END_DATE))

    # Build output lines
    lines = []
    for symbol in sorted(intervals.keys()):
        for start, end in sorted(intervals[symbol]):
            lines.append(f"{symbol}\t{start}\t{end}")

    return lines


def main():
    print("=" * 60)
    print("Creating CSI100 instruments file")
    print("=" * 60)

    # Step 1: Get current constituents
    print("\n[Step 1] Get current CSI100 constituents")
    current = get_current_constituents()

    # Step 2: Get announcement list
    print("\n[Step 2] Get historical change announcements")
    announcements = get_announcement_list()

    # Step 3: Parse each announcement
    print("\n[Step 3] Parse each announcement for CSI100 changes (HTML table index=1)")
    changes = []
    for i, ann in enumerate(announcements):
        print(f"  [{i+1}/{len(announcements)}] {ann['date']} (ID={ann['id']})...", end=" ")
        try:
            eff_date, adds, removes = parse_announcement(ann['id'])
            if eff_date and (adds or removes):
                changes.append((eff_date, adds, removes))
                print(f"effective={eff_date}, +{len(adds)} add, -{len(removes)} remove")
            else:
                print("no CSI100 changes found")
        except Exception as e:
            print(f"error: {e}")

    print(f"\n  Total changes parsed: {len(changes)}")

    # Step 4: Build instruments file
    print("\n[Step 4] Build instruments file")
    lines = build_instruments(current, changes)

    filepath = os.path.join(QLIB_DIR, "csi100.txt")
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    unique_stocks = len(set(l.split('\t')[0] for l in lines))
    print(f"  Written {filepath}")
    print(f"  {unique_stocks} unique stocks, {len(lines)} entries")

    # Verification
    print("\n" + "=" * 60)
    print("Verification:")
    print(f"  Sample entries:")
    for line in lines[:5]:
        print(f"    {line}")
    print(f"  ...")
    for line in lines[-5:]:
        print(f"    {line}")

    # Quick sanity check: count current members
    current_set = set()
    for line in lines:
        parts = line.split('\t')
        if parts[2] == '2026-03-25':
            current_set.add(parts[0])
    print(f"\n  Current members (end_date=2026-03-25): {len(current_set)}")
    print("\nDone!")


if __name__ == '__main__':
    main()

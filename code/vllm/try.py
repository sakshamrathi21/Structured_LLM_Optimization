#!/usr/bin/env python3
"""
parse_throughput.py

Parse throughput lines from a text file and produce a table with columns:
    Qwen-0.5B (r250), Qwen-0.5B (r500), Llama-8B (r250), Llama-8B (r500)

Usage:
    python parse_throughput.py path/to/throughput.txt
    python parse_throughput.py path/to/throughput.txt --out-dir ./results --round 6
"""
from pathlib import Path
import re
import argparse
import pandas as pd
import sys

PATTERN = re.compile(
    r'\(randomness:\s*(\d+)\)\s*\(len factor:\s*([0-9]*\.?[0-9]+)\)\s*\(model_name:\s*([^\)]+)\)\s*:\s*([0-9]+(?:\.[0-9]+)?)',
    re.IGNORECASE
)

DESIRED_COLS = [
    'Qwen-0.5B (r250)',
    'Qwen-0.5B (r500)',
    'Llama-8B (r250)',
    'Llama-8B (r500)'
]

def short_model_name(full: str) -> str:
    s = full.lower()
    if 'qwen' in s:
        return 'Qwen-0.5B'
    if 'llama' in s:
        return 'Llama-8B'
    # fallback: last token
    return full.split('/')[-1].strip()

def parse_file(path: Path):
    text = path.read_text(encoding='utf-8')
    rows = []
    for m in PATTERN.finditer(text):
        randomness = int(m.group(1))
        sys_length = float(m.group(2))*1300
        model_name = m.group(3).strip()
        value = float(m.group(4))
        rows.append({
            'model_name': model_name,
            'randomness': randomness,
            'sys_length': sys_length,
            'throughput': value
        })
    return rows

def build_table(rows):
    if not rows:
        raise SystemExit("No matching lines found in the input file (check formatting).")
    df = pd.DataFrame(rows)
    df['model_short'] = df['model_name'].apply(short_model_name)
    df['col'] = df.apply(lambda r: f"{r['model_short']} (r{r['randomness']})", axis=1)

    df_wide = df.pivot_table(index='sys_length', columns='col', values='throughput', aggfunc='first')

    # Ensure desired columns exist (fill with NaN if missing)
    for c in DESIRED_COLS:
        if c not in df_wide.columns:
            df_wide[c] = pd.NA

    df_wide = df_wide[DESIRED_COLS]
    df_wide = df_wide.sort_index()
    return df_wide

def main():
    parser = argparse.ArgumentParser(description="Parse throughput log into a 4-column table.")
    # parser.add_argument('input', type=Path, help="Input text file with throughput lines.")
    parser.add_argument('--out-dir', type=Path, default=Path('.'), help="Directory to write outputs.")
    parser.add_argument('--csv', default='throughput_table.csv', help="CSV filename (in out-dir).")
    parser.add_argument('--md', default='throughput_table.md', help="Markdown filename (in out-dir).")
    parser.add_argument('--round', type=int, default=6, help="Decimal places to round numeric outputs.")
    args = parser.parse_args()

    # if not args.input.exists():
    #     print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
    #     sys.exit(2)

    rows = parse_file(Path("/home/saksham/CS492/code/vllm/vllm_multiple_families_varying_workloads_output.txt"))
    df_wide = build_table(rows)

    # Round numeric values for presentation (leave NaN as-is)
    try:
        df_out = df_wide.astype(float).round(args.round)
    except Exception:
        df_out = df_wide.copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / args.csv
    out_md = args.out_dir / args.md

    df_out.to_csv(out_csv, index_label='sys_length')

    # Create markdown table
    try:
        md_table = df_out.reset_index().to_markdown(index=False)
    except Exception:
        # fallback simple markdown
        header = ['sys_length'] + list(df_out.columns)
        lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---']*len(header)) + ' |']
        for idx, row in df_out.reset_index().iterrows():
            vals = [str(row['sys_length'])] + [("" if pd.isna(row[c]) else str(row[c])) for c in df_out.columns]
            lines.append('| ' + ' | '.join(vals) + ' |')
        md_table = '\n'.join(lines)

    md_content = "# Throughput Table\n\n" + md_table + "\n"
    out_md.write_text(md_content, encoding='utf-8')

    print(f"Wrote CSV -> {out_csv.resolve()}")
    print(f"Wrote MD  -> {out_md.resolve()}")
    print("\nPreview:\n")
    print(df_out)

if __name__ == '__main__':
    main()

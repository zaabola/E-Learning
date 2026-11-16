"""
Exploratory Data Analysis & Affective Calibration Extractor
Dataset: ASSISTments 2012-2013 with Affect Predictions (2.87 GB, 35 columns)
----------------------------------------------------------------------------
Streams the entire multi-gigabyte dataset in memory-safe chunks to:
  1. Audit all 35 column names, data types, exact null counts, and null %.
  2. Detect behavioral & affective anomalies (negative latencies, spam retries, etc.).
  3. Extract empirical conditional distributions of student affective states:
       - Average_confidence(CONCENTRATING)
       - Average_confidence(FRUSTRATED)
       - Average_confidence(CONFUSED)
       - Average_confidence(BORED)
     across (correctness x latency_zone x hint_used x retry_zone).
  4. Register outputs in `Models/RL/eda_results/`:
       - eda_affective_report.txt
       - affective_calibration.json  (loaded directly by KnowledgeQuestStudentEnv)
       - eda_affective_visualizations.png
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


AFFECT_COLS = [
    'Average_confidence(CONCENTRATING)',
    'Average_confidence(FRUSTRATED)',
    'Average_confidence(CONFUSED)',
    'Average_confidence(BORED)',
]
AFFECT_SHORT = {
    'Average_confidence(CONCENTRATING)': 'concentrating',
    'Average_confidence(FRUSTRATED)': 'frustrated',
    'Average_confidence(CONFUSED)': 'confused',
    'Average_confidence(BORED)': 'bored',
}


def run_affective_eda(csv_path, output_dir, chunksize=300_000):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'eda_affective_report.txt')
    calib_json_path = os.path.join(output_dir, 'affective_calibration.json')
    chart_png_path = os.path.join(output_dir, 'eda_affective_visualizations.png')

    logger = DualLogger(report_path)
    sys.stdout = logger
    t_start = time.time()

    print("=" * 90)
    print("   FORENSIC AFFECTIVE & BEHAVIORAL EDA - ASSISTMENTS 2012-2013 (RL CALIBRATION)   ")
    print("=" * 90)
    print(f"Dataset Path : {csv_path}")
    print(f"Output Folder: {output_dir}\n")

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.stdout = logger.terminal
        logger.close()
        return

    # Inspect header & sample dtypes first
    sample_df = pd.read_csv(csv_path, nrows=500, encoding='latin1', low_memory=False)
    all_columns = list(sample_df.columns)
    inferred_dtypes = {c: str(sample_df[c].dtype) for c in all_columns}

    total_rows = 0
    null_counts = {c: 0 for c in all_columns}

    # Anomaly counters
    non_binary_correct = 0
    neg_rt = 0
    zero_rt = 0
    over_1hr_rt = 0
    zero_attempts = 0
    over_10_attempts = 0
    max_attempts = 0
    hints_used_rows = 0
    bottom_hint_rows = 0
    valid_affect_rows = 0

    # Accumulators for global & conditional affective means
    global_affect_sum = np.zeros(4, dtype=np.float64)
    global_affect_sq = np.zeros(4, dtype=np.float64)

    # Conditional buckets: key = f"c{corr}_rt_{rt_zone}_h{hint_flag}_r{retry_flag}"
    bucket_counts = {}
    bucket_sums = {}

    # Visual chart accumulators
    by_outcome_counts = {'correct_no_hint': 0, 'correct_with_hint': 0, 'wrong_1_try': 0, 'wrong_multi_retry': 0}
    by_outcome_sums = {k: np.zeros(4, dtype=np.float64) for k in by_outcome_counts}

    rt_zones = ['rapid_<5s', 'flow_5_15s', 'deliberate_15_60s', 'struggle_>60s']
    by_rt_counts = {z: 0 for z in rt_zones}
    by_rt_sums = {z: np.zeros(4, dtype=np.float64) for z in rt_zones}

    retry_bins = ['0 (Hint 1st)', '1 Try', '2 Tries', '3-4 Tries', '5+ Tries']
    by_retry_counts = {b: 0 for b in retry_bins}
    by_retry_sums = {b: np.zeros(4, dtype=np.float64) for b in retry_bins}

    print(f"[1/4] Streaming {os.path.getsize(csv_path)/(1024**3):.2f} GB CSV in chunks of {chunksize:,} rows...")

    chunk_idx = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, encoding='latin1', low_memory=False):
        chunk_idx += 1
        n = len(chunk)
        total_rows += n

        # Null audit across all 35 columns
        c_nulls = chunk.isnull().sum()
        for col in all_columns:
            null_counts[col] += int(c_nulls[col])

        # Behavioral anomaly checks
        corr_num = pd.to_numeric(chunk['correct'], errors='coerce')
        non_binary_correct += int((~corr_num.isin([0, 1, 0.0, 1.0])).sum())

        rt_num = pd.to_numeric(chunk['ms_first_response'], errors='coerce')
        neg_rt += int((rt_num < 0).sum())
        zero_rt += int((rt_num == 0).sum())
        over_1hr_rt += int((rt_num > 3_600_000).sum())

        att_num = pd.to_numeric(chunk['attempt_count'], errors='coerce').fillna(1)
        zero_attempts += int((att_num == 0).sum())
        over_10_attempts += int((att_num > 10).sum())
        c_max_att = int(att_num.max()) if len(att_num) > 0 else 0
        if c_max_att > max_attempts:
            max_attempts = c_max_att

        hint_num = pd.to_numeric(chunk['hint_count'], errors='coerce').fillna(0)
        hints_used_rows += int((hint_num > 0).sum())
        bh_num = pd.to_numeric(chunk['bottom_hint'], errors='coerce').fillna(0)
        bottom_hint_rows += int((bh_num > 0).sum())

        # Extract valid affective subset
        aff_df = chunk[AFFECT_COLS].apply(pd.to_numeric, errors='coerce')
        valid_mask = aff_df.notnull().all(axis=1) & corr_num.isin([0, 1, 0.0, 1.0])
        if valid_mask.sum() == 0:
            continue

        v_aff = aff_df[valid_mask].to_numpy(dtype=np.float64)
        v_aff = np.clip(v_aff, 0.0, 1.0)
        v_corr = corr_num[valid_mask].to_numpy(dtype=np.int64)
        v_sec = np.maximum(0.0, rt_num[valid_mask].fillna(20000.0).to_numpy(dtype=np.float64) / 1000.0)
        v_att = att_num[valid_mask].to_numpy(dtype=np.int64)
        v_hint = (hint_num[valid_mask].to_numpy(dtype=np.int64) > 0).astype(np.int64)

        valid_affect_rows += len(v_corr)
        global_affect_sum += v_aff.sum(axis=0)
        global_affect_sq += (v_aff ** 2).sum(axis=0)

        # Latency zone assignment: 0=rapid, 1=flow, 2=deliberate, 3=struggle
        rt_code = np.zeros(len(v_sec), dtype=np.int64)
        rt_code[(v_sec >= 5.0) & (v_sec <= 15.0)] = 1
        rt_code[(v_sec > 15.0) & (v_sec <= 60.0)] = 2
        rt_code[v_sec > 60.0] = 3

        for z_idx, z_name in enumerate(rt_zones):
            m = (rt_code == z_idx)
            cnt = int(m.sum())
            if cnt > 0:
                by_rt_counts[z_name] += cnt
                by_rt_sums[z_name] += v_aff[m].sum(axis=0)

        # Outcome categories
        m_c_noh = (v_corr == 1) & (v_hint == 0)
        m_c_wh = (v_corr == 1) & (v_hint == 1)
        m_w_1 = (v_corr == 0) & (v_att <= 1)
        m_w_m = (v_corr == 0) & (v_att >= 2)
        for key, m in [
            ('correct_no_hint', m_c_noh),
            ('correct_with_hint', m_c_wh),
            ('wrong_1_try', m_w_1),
            ('wrong_multi_retry', m_w_m),
        ]:
            cnt = int(m.sum())
            if cnt > 0:
                by_outcome_counts[key] += cnt
                by_outcome_sums[key] += v_aff[m].sum(axis=0)

        # Retry bins
        for b_label, m in [
            ('0 (Hint 1st)', v_att == 0),
            ('1 Try', v_att == 1),
            ('2 Tries', v_att == 2),
            ('3-4 Tries', (v_att >= 3) & (v_att <= 4)),
            ('5+ Tries', v_att >= 5),
        ]:
            cnt = int(m.sum())
            if cnt > 0:
                by_retry_counts[b_label] += cnt
                by_retry_sums[b_label] += v_aff[m].sum(axis=0)

        # Fine-grained 32-cell calibration grid: (corr x rt_code x hint x multi_retry)
        v_multi = (v_att >= 2).astype(np.int64)
        cell_ids = v_corr * 16 + rt_code * 4 + v_hint * 2 + v_multi
        for cid in np.unique(cell_ids):
            m = (cell_ids == cid)
            c_bit = int((cid >> 4) & 1)
            rt_bits = int((cid >> 2) & 3)
            h_bit = int((cid >> 1) & 1)
            r_bit = int(cid & 1)
            b_key = f"c{c_bit}_{rt_zones[rt_bits]}_h{h_bit}_r{r_bit}"
            bucket_counts[b_key] = bucket_counts.get(b_key, 0) + int(m.sum())
            if b_key not in bucket_sums:
                bucket_sums[b_key] = np.zeros(4, dtype=np.float64)
            bucket_sums[b_key] += v_aff[m].sum(axis=0)

        if chunk_idx % 5 == 0:
            print(f"      ... processed {total_rows:,} rows ({time.time() - t_start:.1f}s)")

    print(f"      Finished streaming {total_rows:,} total rows in {time.time() - t_start:.1f}s.\n")

    # 2. Print 35-Column Schema & Null Audit
    print("[2/4] Complete 35-Column Schema & Missing / Null Audit:")
    print(f"{'#':<3} | {'Column Name':<36} | {'Type':<8} | {'Null Count':<11} | {'Null %':<7}")
    print("-" * 78)
    schema_list = []
    for idx, col in enumerate(all_columns, start=1):
        nc = null_counts[col]
        npct = (nc / max(total_rows, 1)) * 100.0
        print(f"{idx:<3} | {col:<36} | {inferred_dtypes[col]:<8} | {nc:<11,d} | {npct:>6.2f}%")
        schema_list.append({
            'index': idx,
            'column': col,
            'dtype': inferred_dtypes[col],
            'null_count': nc,
            'null_pct': round(npct, 3),
        })
    print()

    # 3. Print Anomaly & Affective Summary
    print("[3/4] Behavioral Hygiene & Affective State Distributions:")
    print(f"  - Total Raw Rows                   : {total_rows:,}")
    print(f"  - Valid Binary + Affective Rows    : {valid_affect_rows:,} ({valid_affect_rows/max(total_rows,1)*100:.2f}%)")
    print(f"  - Non-Binary Correctness Rows      : {non_binary_correct:,}")
    print(f"  - Negative Response Time (<0 ms)   : {neg_rt:,}")
    print(f"  - Zero Response Time (0 ms)        : {zero_rt:,}")
    print(f"  - Abandoned Tabs (>1 hour)         : {over_1hr_rt:,}")
    print(f"  - Zero-Attempt Rows (Hint 1st)     : {zero_attempts:,}")
    print(f"  - Spam Retries (>10 attempts)      : {over_10_attempts:,} (Max = {max_attempts:,})")
    print(f"  - Interactions Using Hints         : {hints_used_rows:,} ({hints_used_rows/max(total_rows,1)*100:.2f}%)")
    print(f"  - Bottom-Out Hint Taken            : {bottom_hint_rows:,} ({bottom_hint_rows/max(total_rows,1)*100:.2f}%)\n")

    global_means = global_affect_sum / max(valid_affect_rows, 1)
    global_stds = np.sqrt(np.maximum(0.0, (global_affect_sq / max(valid_affect_rows, 1)) - global_means ** 2))

    print("--- GLOBAL EMPIRICAL AFFECTIVE CONFIDENCE (N = {:,}) ---".format(valid_affect_rows))
    global_affect_dict = {}
    for i, col in enumerate(AFFECT_COLS):
        short = AFFECT_SHORT[col]
        print(f"  {short.upper():<15} : Mean = {global_means[i]:.4f} | Std = {global_stds[i]:.4f}")
        global_affect_dict[short] = {
            'mean': round(float(global_means[i]), 5),
            'std': round(float(global_stds[i]), 5),
        }

    print("\n--- AFFECTIVE STATE BY STUDENT OUTCOME ---")
    print(f"{'Outcome Category':<24} | {'Rows':<11} | {'Concentrating':<13} | {'Frustrated':<11} | {'Confused':<10} | {'Bored':<8}")
    print("-" * 88)
    outcome_summary = {}
    for key, cnt in by_outcome_counts.items():
        m = by_outcome_sums[key] / max(cnt, 1)
        print(f"{key:<24} | {cnt:<11,d} | {m[0]:<13.4f} | {m[1]:<11.4f} | {m[2]:<10.4f} | {m[3]:<8.4f}")
        outcome_summary[key] = {
            'count': cnt,
            'concentrating': round(float(m[0]), 5),
            'frustrated': round(float(m[1]), 5),
            'confused': round(float(m[2]), 5),
            'bored': round(float(m[3]), 5),
        }

    print("\n--- AFFECTIVE STATE BY RESPONSE LATENCY ZONE ---")
    print(f"{'Latency Zone':<24} | {'Rows':<11} | {'Concentrating':<13} | {'Frustrated':<11} | {'Confused':<10} | {'Bored':<8}")
    print("-" * 88)
    rt_summary = {}
    for z in rt_zones:
        cnt = by_rt_counts[z]
        m = by_rt_sums[z] / max(cnt, 1)
        print(f"{z:<24} | {cnt:<11,d} | {m[0]:<13.4f} | {m[1]:<11.4f} | {m[2]:<10.4f} | {m[3]:<8.4f}")
        rt_summary[z] = {
            'count': cnt,
            'concentrating': round(float(m[0]), 5),
            'frustrated': round(float(m[1]), 5),
            'confused': round(float(m[2]), 5),
            'bored': round(float(m[3]), 5),
        }

    # Build 32-cell calibration lookup for RL StudentEnv
    calibration_cells = {}
    for b_key, cnt in sorted(bucket_counts.items()):
        m = bucket_sums[b_key] / max(cnt, 1)
        calibration_cells[b_key] = {
            'count': cnt,
            'concentrating': round(float(m[0]), 5),
            'frustrated': round(float(m[1]), 5),
            'confused': round(float(m[2]), 5),
            'bored': round(float(m[3]), 5),
        }

    # 4. Generate 4-Panel Visualization
    print("\n[4/4] Generating visualizations and exporting affective_calibration.json...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel A: Global Affective Means
    labels_a = ['Concentrating', 'Frustrated', 'Confused', 'Bored']
    colors_a = ['#059669', '#e11d48', '#d97706', '#6366f1']
    bars_a = axs[0, 0].bar(labels_a, global_means * 100, color=colors_a, edgecolor='black', alpha=0.85)
    axs[0, 0].set_title('A. Global Mean Affective Confidence (%)', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('Mean Detector Confidence (%)')
    axs[0, 0].set_ylim(0, 80)
    for b in bars_a:
        h = b.get_height()
        axs[0, 0].text(b.get_x() + b.get_width()/2, h + 1.5, f'{h:.1f}%', ha='center', fontweight='bold')
    axs[0, 0].grid(axis='y', alpha=0.3)

    # Panel B: Affect by Outcome Category
    out_keys = list(outcome_summary.keys())
    x_b = np.arange(len(out_keys))
    w = 0.25
    axs[0, 1].bar(x_b - w, [outcome_summary[k]['concentrating']*100 for k in out_keys], width=w, label='Concentrating', color='#059669')
    axs[0, 1].bar(x_b,     [outcome_summary[k]['frustrated']*100 for k in out_keys],    width=w, label='Frustrated',    color='#e11d48')
    axs[0, 1].bar(x_b + w, [outcome_summary[k]['bored']*100 for k in out_keys],         width=w, label='Bored',         color='#6366f1')
    axs[0, 1].set_xticks(x_b)
    axs[0, 1].set_xticklabels(['Correct\n(No Hint)', 'Correct\n(With Hint)', 'Wrong\n(1 Try)', 'Wrong\n(Multi-Retry)'], fontsize=9)
    axs[0, 1].set_title('B. Student Affect by Battle / Problem Outcome', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('Mean Confidence (%)')
    axs[0, 1].legend()
    axs[0, 1].grid(axis='y', alpha=0.3)

    # Panel C: Affect Across Response Latency Zones
    x_c = np.arange(len(rt_zones))
    axs[1, 0].plot(x_c, [rt_summary[z]['concentrating']*100 for z in rt_zones], marker='o', linewidth=2.5, color='#059669', label='Concentrating')
    axs[1, 0].plot(x_c, [rt_summary[z]['frustrated']*100 for z in rt_zones],    marker='s', linewidth=2.5, color='#e11d48', label='Frustrated')
    axs[1, 0].plot(x_c, [rt_summary[z]['bored']*100 for z in rt_zones],         marker='^', linewidth=2.5, color='#6366f1', label='Bored')
    axs[1, 0].plot(x_c, [rt_summary[z]['confused']*100 for z in rt_zones],      marker='d', linewidth=2.0, color='#d97706', label='Confused')
    axs[1, 0].set_xticks(x_c)
    axs[1, 0].set_xticklabels(['<5s\n(Rapid)', '5-15s\n(Flow)', '15-60s\n(Deliberate)', '>60s\n(Struggle)'])
    axs[1, 0].set_title('C. Response Time Latency vs. Student Affective State', fontsize=12, fontweight='bold')
    axs[1, 0].set_ylabel('Mean Confidence (%)')
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)

    # Panel D: Retry Persistence vs Frustration / Confusion / Boredom
    x_d = np.arange(len(retry_bins))
    r_frust = [(by_retry_sums[b][1] / max(by_retry_counts[b], 1)) * 100 for b in retry_bins]
    r_conf = [(by_retry_sums[b][2] / max(by_retry_counts[b], 1)) * 100 for b in retry_bins]
    r_conc = [(by_retry_sums[b][0] / max(by_retry_counts[b], 1)) * 100 for b in retry_bins]
    axs[1, 1].bar(x_d - 0.25, r_conc,  width=0.25, color='#059669', label='Concentrating')
    axs[1, 1].bar(x_d,        r_frust, width=0.25, color='#e11d48', label='Frustrated')
    axs[1, 1].bar(x_d + 0.25, r_conf,  width=0.25, color='#d97706', label='Confused')
    axs[1, 1].set_xticks(x_d)
    axs[1, 1].set_xticklabels(retry_bins, fontsize=9)
    axs[1, 1].set_title('D. Attempt / Retry Count vs. Affective Response', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('Mean Confidence (%)')
    axs[1, 1].legend()
    axs[1, 1].grid(axis='y', alpha=0.3)

    plt.savefig(chart_png_path, dpi=200, bbox_inches='tight')
    plt.close()

    calib_payload = {
        'dataset': os.path.basename(csv_path),
        'total_raw_rows': total_rows,
        'valid_affect_rows': valid_affect_rows,
        'anomalies': {
            'non_binary_correct': non_binary_correct,
            'negative_response_time': neg_rt,
            'zero_response_time': zero_rt,
            'over_1hr_response_time': over_1hr_rt,
            'zero_attempts_hint_first': zero_attempts,
            'over_10_attempts': over_10_attempts,
            'max_attempts': max_attempts,
            'hints_used_rows': hints_used_rows,
            'bottom_hint_rows': bottom_hint_rows,
        },
        'global_affect': global_affect_dict,
        'by_outcome': outcome_summary,
        'by_latency_zone': rt_summary,
        'calibration_grid': calibration_cells,
        'schema_audit': schema_list,
    }
    with open(calib_json_path, 'w', encoding='utf-8') as f:
        json.dump(calib_payload, f, indent=2)

    print(f"\n[SUCCESS] Registered all RL Affective EDA outputs in: {output_dir}")
    print(f"  1. Text Report           -> {report_path}")
    print(f"  2. RL Calibration JSON   -> {calib_json_path}")
    print(f"  3. Visual Dashboard PNG  -> {chart_png_path}")
    print("=" * 90)

    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    rl_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.dirname(rl_dir)
    default_csv = os.path.join(models_dir, 'DKT', 'data', '2012-2013-data-with-predictions-4-final.csv')
    default_out = os.path.join(rl_dir, 'eda_results')
    run_affective_eda(default_csv, default_out)

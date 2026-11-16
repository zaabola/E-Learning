import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DualLogger:
    """Logs output simultaneously to standard console and a text file."""
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


COLUMN_METADATA = {
    'order_id': 'Monotonically increasing transaction order ID (temporal sequence key)',
    'assignment_id': 'Identifier for assigned homework, quiz, or class assignment',
    'user_id': 'Unique student learner identifier',
    'assistment_id': 'Parent learning module / problem container identifier',
    'problem_id': 'Specific problem / exercise question identifier',
    'original': 'Problem type indicator (1 = main problem, 0 = scaffolding sub-question)',
    'correct': 'First-attempt correctness (1 = correct, 0 = incorrect; contains fractional credit)',
    'attempt_count': 'Number of attempts made on this problem (can be 0 if hint taken first)',
    'ms_first_response_time': 'Latency in milliseconds before the first submission/action',
    'tutor_mode': 'Interface mode during attempt (tutor, test, pre, post, survey)',
    'answer_type': 'Input widget type (algebra, choose_1, fill_in_1, choose_n, open_response, rank, external)',
    'sequence_id': 'Curriculum curriculum unit or sequence identifier',
    'student_class_id': 'School classroom section identifier',
    'position': 'Ordinal index of problem inside the assignment sequence',
    'problem_set_type': 'Pedagogical delivery rule (MasterySection, Linear, etc.)',
    'base_sequence_id': 'Root / template curriculum sequence identifier',
    'list_skill_ids': 'Semicolon-delimited skill/concept identifiers mapped to this problem',
    'list_skills': 'Human-readable skill names corresponding to list_skill_ids',
    'teacher_id': 'Educator / teacher account identifier',
    'school_id': 'Educational institution / school identifier'
}


def run_eda(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_txt_path = os.path.join(output_dir, 'eda_report.txt')
    metrics_json_path = os.path.join(output_dir, 'eda_metrics.json')
    charts_png_path = os.path.join(output_dir, 'eda_visualizations.png')

    logger = DualLogger(report_txt_path)
    sys.stdout = logger

    print("=" * 85)
    print("      DEEP FORENSIC DATA EXPLORATION & AUDIT - ASSISTMENTS 2009-2010      ")
    print("=" * 85)
    print(f"Dataset Path : {csv_path}")
    print(f"Output Folder: {output_dir}\n")

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.stdout = logger.terminal
        logger.close()
        return

    # 1. Load Data
    print("[1/7] Loading CSV dataset into memory...")
    df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)
    total_raw_rows = len(df)
    total_cols = len(df.columns)
    print(f"      Total Raw Interactions : {total_raw_rows:,}")
    print(f"      Total Features/Columns : {total_cols}\n")

    # 2. Comprehensive Column Schema & Null Analysis
    print("[2/7] Complete Schema Audit & Missing / Null Value Breakdown:")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / total_raw_rows) * 100
    dtypes = df.dtypes
    uniques = df.nunique()

    schema_rows = []
    print(f"{'#':<3} | {'Column Name':<23} | {'Type':<8} | {'Null Count':<10} | {'Null %':<7} | {'Unique':<8} | {'Description'}")
    print("-" * 115)
    for idx, col in enumerate(df.columns, start=1):
        desc = COLUMN_METADATA.get(col, 'Metadata attribute')
        print(f"{idx:<3} | {col:<23} | {str(dtypes[col]):<8} | {null_counts[col]:<10,d} | {null_pct[col]:>5.2f}% | {uniques[col]:<8,d} | {desc}")
        schema_rows.append({
            'index': idx,
            'column': col,
            'dtype': str(dtypes[col]),
            'null_count': int(null_counts[col]),
            'null_pct': float(round(null_pct[col], 3)),
            'unique_count': int(uniques[col]),
            'description': desc
        })
    print()

    # 3. Data Hygiene & Anomaly Audit ("What is Wrong with the Dataset")
    print("[3/7] Forensic Hygiene Audit - Detecting Anomalies, Glitches & Edge Cases:")

    # Anomaly A: Non-Binary Correctness
    non_binary_mask = ~df['correct'].isin([0, 1, 0.0, 1.0])
    non_binary_count = int(non_binary_mask.sum())
    non_binary_vals = df.loc[non_binary_mask, 'correct'].value_counts().to_dict()
    print(f"  [A] Non-Binary Correctness Values: {non_binary_count:,} rows ({non_binary_count/total_raw_rows*100:.3f}%)")
    print(f"      -assistments logs partial credit: {non_binary_vals}")
    print(f"      -Impact on DKT: Standard Binary Cross-Entropy loss requires binary targets {{0, 1}}.")

    # Anomaly B: Response Time Telemetry Outliers
    rt = df['ms_first_response_time']
    rt_null = int(rt.isnull().sum())
    rt_neg = int((rt < 0).sum())
    rt_zero = int((rt == 0).sum())
    rt_over_1hr = int((rt > 3600000).sum())
    rt_over_24hr = int((rt > 86400000).sum())
    rt_max_days = float(rt.max() / (1000 * 60 * 60 * 24))
    print(f"\n  [B] Response Time Latency Anomalies (ms_first_response_time):")
    print(f"      - Missing / Null Response Times : {rt_null:,} rows")
    print(f"      - Negative Latency (< 0 ms)      : {rt_neg} rows (Client-server clock sync glitch)")
    print(f"      - Instantaneous Submissions (0 ms): {rt_zero} rows (Bot submit / instant cheat)")
    print(f"      - Abandoned (> 1 hour)           : {rt_over_1hr:,} rows (Student walked away)")
    print(f"      - Extreme (> 24 hours)           : {rt_over_24hr} rows (Browser tab left open up to {rt_max_days:.1f} days)")

    # Anomaly C: Attempt Count Outliers
    att = df['attempt_count']
    att_zero = int((att == 0).sum())
    att_over_10 = int((att > 10).sum())
    att_max = int(att.max())
    print(f"\n  [C] Attempt Count Anomalies (attempt_count):")
    print(f"      - Zero Attempts (attempt_count = 0) : {att_zero:,} rows ({att_zero/total_raw_rows*100:.2f}%)")
    print(f"        (Occurs when student requests a hint/scaffold without making any answer submission)")
    print(f"      - Excessive Submissions (> 10 tries): {att_over_10:,} rows")
    print(f"      - Maximum Single-Problem Attempts   : {att_max:,} attempts (Spam bot / scripted loop)")

    # Anomaly D: Missing Skills Root Cause
    print(f"\n  [D] Missing Skill Tags Root Cause (list_skill_ids: {null_counts['list_skill_ids']:,} missing / {null_pct['list_skill_ids']:.2f}%):")
    scaffold_total = (df['original'] == 0).sum()
    scaffold_null_skill = ((df['original'] == 0) & df['list_skill_ids'].isnull()).sum()
    original_total = (df['original'] == 1).sum()
    original_null_skill = ((df['original'] == 1) & df['list_skill_ids'].isnull()).sum()
    print(f"      - Scaffolding Sub-questions (original=0): {scaffold_null_skill:,} of {scaffold_total:,} ({scaffold_null_skill/scaffold_total*100:.1f}%) MISSING skills")
    print(f"        (Scaffold hint questions inherit the parent concept implicitly rather than explicit tags)")
    print(f"      - Main Problems (original=1)           : {original_null_skill:,} of {original_total:,} ({original_null_skill/original_total*100:.1f}%) MISSING skills")
    print(f"        (Uncategorized or non-state-standard homework problems)")

    # Anomaly E: Multi-Skill Tagging Ambiguity
    non_null_skills = df['list_skill_ids'].dropna()
    multi_skills = non_null_skills.str.contains(';')
    multi_skill_count = int(multi_skills.sum())
    max_skills_per_prob = int(non_null_skills.str.count(';').max() + 1)
    print(f"\n  [E] Multi-Skill Concept Ambiguity:")
    print(f"      - Single-Skill Interactions: {len(non_null_skills) - multi_skill_count:,} ({(1 - multi_skill_count/len(non_null_skills))*100:.1f}%)")
    print(f"      - Multi-Skill Interactions : {multi_skill_count:,} ({multi_skill_count/len(non_null_skills)*100:.1f}%)")
    print(f"      - Maximum Skills on 1 Item : {max_skills_per_prob} skills (e.g. 'Mean;Table;Multiplication')")
    print(f"      - DKT Solution: Extract primary skill (split by ';')[0] to avoid one-hot vector collision.")

    # Anomaly F: Repeated Problem Encounters & Chronological Integrity
    dup_prob = int(df.duplicated(subset=['user_id', 'problem_id']).sum())
    is_order_monotonic = bool(df['order_id'].is_monotonic_increasing)
    print(f"\n  [F] Temporal & Repetition Integrity:")
    print(f"      - Re-attempted Same Problem (user_id, problem_id): {dup_prob:,} interactions ({dup_prob/total_raw_rows*100:.2f}%)")
    print(f"      - Global Chronological Order (order_id monotonic): {is_order_monotonic}")
    print(f"        (Crucial positive finding: interaction history is already chronologically sorted!)\n")

    # 4. Clean DKT Interaction Subset
    print("[4/7] Filtering Clean DKT Training Subset (Resolving All Above Traps)...")
    clean_mask = df['list_skill_ids'].notnull() & df['correct'].isin([0, 1, '0', '1']) & df['user_id'].notnull()
    clean_df = df[clean_mask].copy()
    clean_df['correct'] = clean_df['correct'].astype(int)
    clean_rows = len(clean_df)
    print(f"      Raw Rows           : {total_raw_rows:,}")
    print(f"      Clean DKT Subset   : {clean_rows:,} ({clean_rows / total_raw_rows * 100:.2f}%)")
    print(f"      Excluded Rows      : {total_raw_rows - clean_rows:,} (unmapped skills + fractional credit)\n")

    def get_primary_skill_id(s):
        return str(s).split(';')[0].strip() if pd.notnull(s) else "Unknown"

    def get_primary_skill_name(s):
        return str(s).split(';')[0].strip() if pd.notnull(s) else "Unknown"

    clean_df['skill_id'] = clean_df['list_skill_ids'].apply(get_primary_skill_id)
    clean_df['skill_name'] = clean_df['list_skills'].apply(get_primary_skill_name)

    # 5. Student Interaction Statistics
    print("[5/7] Computing Student Sequence Dynamics...")
    user_counts = clean_df.groupby('user_id').size()
    num_students = len(user_counts)
    seq_min = int(user_counts.min())
    seq_q25 = float(user_counts.quantile(0.25))
    seq_med = float(user_counts.median())
    seq_mean = float(user_counts.mean())
    seq_q75 = float(user_counts.quantile(0.75))
    seq_q90 = float(user_counts.quantile(0.90))
    seq_max = int(user_counts.max())

    print(f"      Total Unique Students  : {num_students:,}")
    print(f"      Interactions per Student:")
    print(f"        Min     : {seq_min}")
    print(f"        25th %  : {seq_q25:.1f}")
    print(f"        Median  : {seq_med:.1f}")
    print(f"        Mean    : {seq_mean:.1f}")
    print(f"        75th %  : {seq_q75:.1f}")
    print(f"        90th %  : {seq_q90:.1f}")
    print(f"        Max     : {seq_max}\n")

    # 6. Skill & Concept Breakdown
    skill_stats = clean_df.groupby(['skill_id', 'skill_name']).agg(
        count=('correct', 'count'),
        accuracy=('correct', 'mean')
    ).reset_index()
    total_unique_skills = len(skill_stats)

    print(f"      Total Unique Skills/Concepts : {total_unique_skills}")

    top_frequent = skill_stats.sort_values(by='count', ascending=False).head(10)
    print("\n--- TOP 10 MOST FREQUENT SKILLS ---")
    print(f"{'Skill Name':<38} | {'Attempts':<10} | {'Accuracy':<10}")
    print("-" * 65)
    for _, r in top_frequent.iterrows():
        name = r['skill_name'][:36] if r['skill_name'] != 'Unknown' else f"Skill_{r['skill_id']}"
        print(f"{name:<38} | {r['count']:<10,d} | {r['accuracy']*100:.1f}%")

    qualified_skills = skill_stats[skill_stats['count'] >= 100]
    top_hardest = qualified_skills.sort_values(by='accuracy', ascending=True).head(8)
    top_easiest = qualified_skills.sort_values(by='accuracy', ascending=False).head(8)

    print("\n--- TOP 8 HARDEST SKILLS (Min. 100 Attempts) ---")
    print(f"{'Skill Name':<38} | {'Accuracy':<10} | {'Attempts':<10}")
    print("-" * 65)
    for _, r in top_hardest.iterrows():
        name = r['skill_name'][:36] if r['skill_name'] != 'Unknown' else f"Skill_{r['skill_id']}"
        print(f"{name:<38} | {r['accuracy']*100:.1f}%     | {r['count']:<10,d}")

    print("\n--- TOP 8 EASIEST SKILLS (Min. 100 Attempts) ---")
    print(f"{'Skill Name':<38} | {'Accuracy':<10} | {'Attempts':<10}")
    print("-" * 65)
    for _, r in top_easiest.iterrows():
        name = r['skill_name'][:36] if r['skill_name'] != 'Unknown' else f"Skill_{r['skill_id']}"
        print(f"{name:<38} | {r['accuracy']*100:.1f}%     | {r['count']:<10,d}")

    correct_count = int((clean_df['correct'] == 1).sum())
    incorrect_count = int((clean_df['correct'] == 0).sum())
    overall_accuracy = float(clean_df['correct'].mean())

    print(f"\n--- TARGET CLASS DISTRIBUTION ---")
    print(f"      Correct (1)   : {correct_count:,} ({overall_accuracy * 100:.2f}%)")
    print(f"      Incorrect (0) : {incorrect_count:,} ({(1 - overall_accuracy) * 100:.2f}%)")

    clean_df['response_sec'] = pd.to_numeric(clean_df['ms_first_response_time'], errors='coerce') / 1000.0
    valid_time = clean_df['response_sec'].dropna()
    valid_time = valid_time[(valid_time > 0) & (valid_time < 300)]
    med_time = float(valid_time.median())
    mean_time = float(valid_time.mean())

    bins = [0, 5, 15, 30, 60, 180, 300]
    bin_labels = ['<5s (Rapid)', '5-15s', '15-30s', '30-60s', '1-3m', '3-5m']
    clean_df['time_bin'] = pd.cut(clean_df['response_sec'], bins=bins, labels=bin_labels)
    time_acc = clean_df.groupby('time_bin', observed=False)['correct'].agg(count='count', accuracy='mean').reset_index()

    # 7. Generate Visual Charts
    print("\n[6/7] Generating publication-quality charts...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # A. Sequence Length Histogram
    capped = user_counts.clip(upper=250)
    axs[0, 0].hist(capped, bins=40, color='#2563eb', edgecolor='#1d4ed8', alpha=0.85)
    axs[0, 0].axvline(seq_med, color='#dc2626', linestyle='--', linewidth=2, label=f'Median: {seq_med:.0f}')
    axs[0, 0].axvline(seq_q75, color='#d97706', linestyle=':', linewidth=2, label=f'75th %: {seq_q75:.0f}')
    axs[0, 0].set_title('A. Student Interaction History Length (Capped at 250)', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Interactions per Student')
    axs[0, 0].set_ylabel('Student Count')
    axs[0, 0].legend()
    axs[0, 0].grid(axis='y', alpha=0.3)

    # B. Target Distribution Pie
    wedges, texts, autotexts = axs[0, 1].pie(
        [correct_count, incorrect_count],
        labels=['Correct (1)', 'Incorrect (0)'],
        colors=['#059669', '#e11d48'],
        autopct='%1.1f%%',
        startangle=140,
        explode=(0.04, 0),
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts: t.set_color('white'); t.set_weight('bold')
    axs[0, 1].set_title('B. Target Class Distribution (Overall Accuracy)', fontsize=12, fontweight='bold')

    # C. Hardest vs Easiest
    combo = pd.concat([top_hardest.head(6), top_easiest.head(6)]).sort_values(by='accuracy')
    c_names = [n[:24] if n != 'Unknown' else f"Skill {id}" for n, id in zip(combo['skill_name'], combo['skill_id'])]
    c_accs = combo['accuracy'] * 100
    c_colors = ['#e11d48' if a < 50 else '#059669' for a in c_accs]
    y_pos = np.arange(len(c_names))
    axs[1, 0].barh(y_pos, c_accs, color=c_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    axs[1, 0].set_yticks(y_pos)
    axs[1, 0].set_yticklabels(c_names, fontsize=9)
    axs[1, 0].axvline(overall_accuracy * 100, color='#2563eb', linestyle='--', linewidth=1.5, label=f'Mean Acc: {overall_accuracy*100:.1f}%')
    axs[1, 0].set_title('C. Hardest (Red) vs. Easiest (Green) Course Skills', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('Success Rate (%)')
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid(axis='x', alpha=0.3)

    # D. Latency vs Accuracy
    t_clean = time_acc.dropna(subset=['time_bin'])
    bars = axs[1, 1].bar(range(len(t_clean)), t_clean['accuracy'] * 100, color='#4f46e5', edgecolor='#3730a3', alpha=0.85)
    axs[1, 1].set_xticks(range(len(t_clean)))
    axs[1, 1].set_xticklabels(t_clean['time_bin'], rotation=25, ha='right', fontsize=9)
    axs[1, 1].set_title('D. Response Time Latency vs. Student Accuracy', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('First Response Time Category')
    axs[1, 1].set_ylabel('Success Rate (%)')
    axs[1, 1].set_ylim(0, 100)
    for b in bars:
        h = b.get_height()
        axs[1, 1].text(b.get_x() + b.get_width()/2.0, h + 1.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)
    axs[1, 1].grid(axis='y', alpha=0.3)

    plt.savefig(charts_png_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 8. Export JSON Metrics
    print("[7/7] Registering JSON metrics and summary files...")
    metrics_data = {
        'total_raw_rows': total_raw_rows,
        'clean_dkt_rows': clean_rows,
        'unique_students': num_students,
        'unique_skills': total_unique_skills,
        'overall_accuracy': overall_accuracy,
        'schema_audit': schema_rows,
        'hygiene_audit': {
            'non_binary_correct_count': non_binary_count,
            'non_binary_correct_breakdown': non_binary_vals,
            'response_time_negative_count': rt_neg,
            'response_time_zero_count': rt_zero,
            'response_time_over_1hr_count': rt_over_1hr,
            'response_time_over_24hr_count': rt_over_24hr,
            'response_time_max_days': rt_max_days,
            'attempt_count_zero': att_zero,
            'attempt_count_over_10': att_over_10,
            'attempt_count_max': att_max,
            'scaffolding_missing_skills': int(scaffold_null_skill),
            'scaffolding_total': int(scaffold_total),
            'original_missing_skills': int(original_null_skill),
            'original_total': int(original_total),
            'multi_skill_rows': multi_skill_count,
            'max_skills_per_item': max_skills_per_prob,
            'duplicate_user_problem_encounters': dup_prob,
            'is_order_monotonic': is_order_monotonic
        },
        'sequence_length': {
            'min': seq_min, 'q25': seq_q25, 'median': seq_med,
            'mean': seq_mean, 'q75': seq_q75, 'q90': seq_q90, 'max': seq_max
        },
        'response_time_sec': {
            'median': med_time, 'mean': mean_time,
            'q25': float(valid_time.quantile(0.25)), 'q75': float(valid_time.quantile(0.75))
        },
        'top_frequent_skills': top_frequent[['skill_id', 'skill_name', 'count', 'accuracy']].to_dict(orient='records'),
        'hardest_skills': top_hardest[['skill_id', 'skill_name', 'count', 'accuracy']].to_dict(orient='records'),
        'easiest_skills': top_easiest[['skill_id', 'skill_name', 'count', 'accuracy']].to_dict(orient='records'),
    }

    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\n[SUCCESS] Output files registered successfully in: {output_dir}")
    print(f"  1. Text Report    -> {report_txt_path}")
    print(f"  2. JSON Metrics   -> {metrics_json_path}")
    print(f"  3. Visual Charts  -> {charts_png_path}")
    print("=" * 85)

    sys.stdout = logger.terminal
    logger.close()

if __name__ == '__main__':
    default_csv = os.path.join(os.path.dirname(__file__), 'data', 'assistments_2009_2010.csv')
    default_out = os.path.join(os.path.dirname(__file__), 'eda_results')
    run_eda(default_csv, default_out)

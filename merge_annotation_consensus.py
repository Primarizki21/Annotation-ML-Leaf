import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np

# Tentukan direktori tempat CSV masing-masing anggota tersimpan
# Sesuaikan path ini dengan struktur folder kamu
CSV_DIR = "./annotations/*.csv"
OUTPUT_FILE = "consensus_review_master.csv"

def _fleiss_kappa(df, annotator_cols, n_annotators):
    """Compute Fleiss' Kappa for a dataframe subset."""
    N = len(df)
    n_healthy = np.zeros(N)
    n_unhealthy = np.zeros(N)

    for i, (_, row) in enumerate(df.iterrows()):
        answers = row[annotator_cols].dropna().values
        n_healthy[i] = sum(1 for a in answers if a == 'healthy')
        n_unhealthy[i] = len(answers) - n_healthy[i]

    sum_squares = n_healthy**2 + n_unhealthy**2
    P_i = (sum_squares - n_annotators) / (n_annotators * (n_annotators - 1))
    P_bar = np.mean(P_i)

    p_healthy = np.sum(n_healthy) / (N * n_annotators)
    p_unhealthy = np.sum(n_unhealthy) / (N * n_annotators)
    P_e = p_healthy**2 + p_unhealthy**2

    if P_e >= 1.0:
        return 1.0

    return (P_bar - P_e) / (1 - P_e)


def _interpret_kappa(kappa):
    """Interpret Fleiss' Kappa using Landis & Koch scale."""
    if kappa is None:
        return "N/A"
    if kappa < 0:
        return "Poor agreement (<0)"
    elif kappa <= 0.2:
        return "Slight agreement (0.00-0.20)"
    elif kappa <= 0.4:
        return "Fair agreement (0.21-0.40)"
    elif kappa <= 0.6:
        return "Moderate agreement (0.41-0.60)"
    elif kappa <= 0.8:
        return "Substantial agreement (0.61-0.80)"
    else:
        return "Almost perfect agreement (0.81-1.00)"


def compute_fleiss_kappa(df_pivot, annotator_cols):
    """Compute Fleiss' Kappa for inter-rater agreement.
    Args:
        df_pivot: DataFrame with patch_path, class_name, and annotator columns.
        annotator_cols: List of annotator column names.

    Returns:
        overall_kappa, per_class_kappa, agreement_summary.
    """
    df_complete = df_pivot.dropna(subset=annotator_cols)
    n_annotators = len(annotator_cols)

    if len(df_complete) == 0:
        return None, {}, {}

    kappa_overall = _fleiss_kappa(df_complete, annotator_cols, n_annotators)

    kappa_per_class = {}
    for class_name, group in df_complete.groupby('class_name'):
        if len(group) > 1:
            kappa_per_class[class_name] = _fleiss_kappa(group, annotator_cols, n_annotators)

    agreement_summary = {
        'unanimous (5/5)': 0,
        'strong_majority (4/5)': 0,
        'split (3/2)': 0,
    }
    for _, row in df_complete.iterrows():
        answers = row[annotator_cols].values
        healthy_count = sum(1 for a in answers if a == 'healthy')
        unhealthy_count = sum(1 for a in answers if a == 'unhealthy')
        majority = max(healthy_count, unhealthy_count)

        if majority == 5:
            agreement_summary['unanimous (5/5)'] += 1
        elif majority == 4:
            agreement_summary['strong_majority (4/5)'] += 1
        elif majority == 3:
            agreement_summary['split (3/2)'] += 1

    return kappa_overall, kappa_per_class, agreement_summary


def save_class_fleiss_kappa(kappa_per_class, output_path):
    valid = {k: v for k, v in kappa_per_class.items() if v is not None}
    if not valid:
        print("Skipping per-class Fleiss' Kappa plot: no valid kappa values")
        return

    names = sorted(valid, key=valid.get, reverse=True)
    scores = [valid[n] for n in names]

    colors = []
    for s in scores:
        if s >= 0.81:
            colors.append("#2ecc71")
        elif s >= 0.61:
            colors.append("#3498db")
        elif s >= 0.41:
            colors.append("#f39c12")
        elif s >= 0.21:
            colors.append("#e74c3c")
        else:
            colors.append("#c0392b")

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
    bars = ax.barh(range(len(names)), scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Fleiss' Kappa")
    ax.set_title("Per-Class Fleiss' Kappa (Inter-Rater Agreement)")

    for threshold, label, color in [
        (0.81, "Almost perfect", "#2ecc71"),
        (0.61, "Substantial", "#3498db"),
        (0.41, "Moderate", "#f39c12"),
        (0.21, "Fair", "#e74c3c"),
        (0.00, "Slight / Poor", "#c0392b"),
    ]:
        ax.axvline(threshold, color=color, linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(threshold + 0.005, -0.5, label, color=color, fontsize=6)

    x_min = min(min(scores) - 0.05, -0.05)
    ax.set_xlim(x_min, 1.05)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(min(score, 1.0) + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved per-class Fleiss' Kappa chart: {output_path}")


def generate_consensus_spreadsheet():
    # 1. Baca semua file CSV hasil anotasi
    csv_files = glob.glob(CSV_DIR)
    if not csv_files:
        print("File CSV tidak ditemukan!")
        return
    
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        # Ambil hanya kolom yang relevan
        temp_df = temp_df[['patch_path', 'class_name', 'label', 'annotator', 'is_skipped']]
        df_list.append(temp_df)
    
    # Gabungkan semua data
    df_all = pd.concat(df_list, ignore_index=True)
    
    # Abaikan yang di-skip saat menghitung konsensus (opsional, tergantung kebijakan)
    df_valid = df_all[df_all['is_skipped'] == False]

    # 2. Pivot data: baris = patch, kolom = annotator
    df_pivot = df_valid.pivot_table(
        index=['patch_path', 'class_name'], 
        columns='annotator', 
        values='label', 
        aggfunc='first'
    ).reset_index()

    # -- Fleiss' Kappa & agreement analysis --
    annotator_cols = df_pivot.columns[2:].tolist()
    kappa_overall, kappa_per_class, agreement_summary = compute_fleiss_kappa(df_pivot, annotator_cols)

    # 3. Hitung persetujuan dan konversi ke label numerik
    def calculate_consensus(row):
        # Ambil hanya jawaban dari annotator (mulai dari kolom ke-3)
        answers = row[2:].dropna().tolist()
        
        if not answers:
            return pd.Series([None, 0, True])
        
        # Hitung mayoritas
        majority_text = max(set(answers), key=answers.count)
        agreement_ratio = answers.count(majority_text) / len(answers)
        
        # Flag konflik jika tidak 100% sepakat
        needs_discussion = agreement_ratio <= 0.6
        
        # Konversi ke numerik untuk standar training model (0 = healthy, 1 = unhealthy)
        numeric_label = 0 if majority_text == "healthy" else 1
        
        return pd.Series([numeric_label, f"{answers.count(majority_text)}/{len(answers)}", needs_discussion])

    # Terapkan fungsi ke dataframe
    df_pivot[['suggested_numeric_label', 'agreement_level', 'needs_discussion']] = df_pivot.apply(calculate_consensus, axis=1)

    # Urutkan berdasarkan yang paling butuh diskusi (konflik di atas)
    df_pivot = df_pivot.sort_values(by=['needs_discussion', 'patch_path'], ascending=[False, True])

    # 4. Simpan hasilnya
    df_pivot.to_csv(OUTPUT_FILE, index=False)
    print(f"File berhasil digenerate: {OUTPUT_FILE}")
    print(f"Total patch yang butuh diskusi: {df_pivot['needs_discussion'].sum()}")

    # -- Fleiss' Kappa report --
    if kappa_overall is not None:
        print("\n=== Fleiss' Kappa (Inter-Rater Agreement) ===")
        print(f"Annotators: {len(annotator_cols)} ({', '.join(annotator_cols)})")
        print(f"Patches rated by all annotators: {sum(agreement_summary.values()):,}")
        print(f"Fleiss' Kappa: {kappa_overall:.4f} — {_interpret_kappa(kappa_overall)}")

        total = sum(agreement_summary.values())
        print("\nAgreement Distribution:")
        for level, count in agreement_summary.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {level}: {count:,} ({pct:.1f}%)")

        if kappa_per_class:
            print("\nPer-Class Fleiss' Kappa:")
            for class_name, kappa in sorted(kappa_per_class.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {kappa:.4f} — {_interpret_kappa(kappa)}")
            save_class_fleiss_kappa(kappa_per_class, "fleiss_kappa_per_class.png")
    else:
        print("\nFleiss' Kappa: not computed (no patches with complete annotations).")

if __name__ == "__main__":
    generate_consensus_spreadsheet()
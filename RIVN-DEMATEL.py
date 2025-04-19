# RIVN-DEMATEL Final Python Script (TÜM AŞAMALARLA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- STEP 0: Parameters ---
linguistic_to_svnn = {
    0: (0.1, 0.8, 0.9),
    1: (0.3, 0.6, 0.7),
    2: (0.5, 0.5, 0.5),
    3: (0.7, 0.4, 0.3),
    4: (0.9, 0.2, 0.1),
    "R": (0.0, 1.0, 0.0)
}

# --- STEP 1: Read Excel File ---
def read_excel_file():
    Tk().withdraw()
    file_path = askopenfilename(title="Select expert opinion Excel file", filetypes=[("Excel files", "*.xlsx *.xls")])
    df = pd.read_excel(file_path, header=None)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    factor_count = df.shape[1]
    expert_count = df.shape[0] // factor_count
    expert_opinions = np.array(df).reshape((expert_count, factor_count, factor_count))
    return expert_opinions, expert_count, factor_count

# --- STEP 2: Convert to SVNN ---
def convert_to_svnn(expert_opinions, linguistic_to_svnn):
    expert_count, factor_count, _ = expert_opinions.shape
    svnn_opinions = np.empty((expert_count, factor_count, factor_count, 3), dtype=float)
    for e in range(expert_count):
        for i in range(factor_count):
            for j in range(factor_count):
                val = expert_opinions[e, i, j]
                try:
                    key = int(val)
                    svnn_opinions[e, i, j] = linguistic_to_svnn.get(key, (0.0, 0.0, 1.0))
                except:
                    svnn_opinions[e, i, j] = (0.0, 0.0, 1.0)
    return svnn_opinions

# --- STEP 3: SVNN to IVNN ---
def svnn_to_ivnn(svnn_opinions):
    expert_count, factor_count, _, _ = svnn_opinions.shape
    ivnn_opinions = np.zeros((expert_count, factor_count, factor_count, 3, 2))
    for e in range(expert_count):
        for i in range(factor_count):
            for j in range(factor_count):
                for k in range(3):
                    all_values = svnn_opinions[:, i, j, k]
                    current_value = svnn_opinions[e, i, j, k]
                    lower_vals = [val for val in all_values if val <= current_value]
                    upper_vals = [val for val in all_values if val >= current_value]
                    ivnn_opinions[e, i, j, k, 0] = np.mean(lower_vals) if lower_vals else current_value
                    ivnn_opinions[e, i, j, k, 1] = np.mean(upper_vals) if upper_vals else current_value
    return ivnn_opinions

# --- STEP 4: IVNNWA Aggregation ---
def apply_ivnnwa(ivnn_data):
    expert_count, factor_count, _, _, _ = ivnn_data.shape
    weights = np.full(expert_count, 1 / expert_count)
    aggregated_ivnn = np.zeros((factor_count, factor_count, 3, 2))
    for i in range(factor_count):
        for j in range(factor_count):
            for k in range(3):
                values_L = ivnn_data[:, i, j, k, 0]
                values_U = ivnn_data[:, i, j, k, 1]
                if k == 0:
                    aggregated_ivnn[i, j, k, 0] = 1 - np.prod([(1 - x) ** w for x, w in zip(values_L, weights)])
                    aggregated_ivnn[i, j, k, 1] = 1 - np.prod([(1 - x) ** w for x, w in zip(values_U, weights)])
                else:
                    aggregated_ivnn[i, j, k, 0] = np.prod([x ** w for x, w in zip(values_L, weights)])
                    aggregated_ivnn[i, j, k, 1] = np.prod([x ** w for x, w in zip(values_U, weights)])
    return aggregated_ivnn

# --- STEP 5: Deneutrosophication ---
def deneutrosophication(aggregated_ivnn):
    factor_count = aggregated_ivnn.shape[0]
    crisp_matrix = np.zeros((factor_count, factor_count))
    for i in range(factor_count):
        for j in range(factor_count):
            T_L, T_U = aggregated_ivnn[i, j, 0]
            I_L, I_U = aggregated_ivnn[i, j, 1]
            F_L, F_U = aggregated_ivnn[i, j, 2]
            numerator = (T_L + T_U + (1 - F_L) + (1 - F_U) + (T_L * T_U) + np.sqrt(abs((1 - F_L) * (1 - F_U)))) / 6
            denominator = ((1 - (I_L + I_U) / 2) * np.sqrt(abs((1 - I_L) * (1 - I_U)))) / 2
            crisp_matrix[i, j] = numerator * denominator if denominator != 0 else 0
    return crisp_matrix

# --- STEP 6: Normalize Crisp Matrix ---
def normalize_crisp_matrix(crisp_matrix):
    row_sums = np.sum(crisp_matrix, axis=1)
    col_sums = np.sum(crisp_matrix, axis=0)
    max_row_sum = np.max(row_sums)
    max_col_sum = np.max(col_sums)
    k = min(1 / max_row_sum, 1 / max_col_sum) if max(max_row_sum, max_col_sum) != 0 else 0
    return crisp_matrix * k, k

# --- STEP 7: Compute Total Relation Matrix ---
def compute_total_relation_matrix(normalized_matrix):
    factor_count = normalized_matrix.shape[0]
    identity_matrix = np.identity(factor_count)
    try:
        inverse_part = np.linalg.inv(identity_matrix - normalized_matrix)
        return normalized_matrix @ inverse_part
    except np.linalg.LinAlgError:
        print("❌ Matrix (I - N) is not invertible.")
        return np.zeros_like(normalized_matrix)

# --- STEP 8: D, R, D+R, D-R ---
def compute_prominence_and_relation(T):
    D = np.sum(T, axis=1)
    R = np.sum(T, axis=0)
    return D, R, D + R, D - R




# --- STEP 10: DEMATEL Cause-Effect Diagram with Color Groups ---
def plot_dematel_cause_effect(D, R, filename="DEMATEL_Cause_Effect.png"):
    prominence = D + R
    relation = D - R
    factor_count = len(D)

    plt.figure(figsize=(10, 8))

    for i in range(factor_count):
        x = prominence[i]
        y = relation[i]
        color = 'red' if y > 0 else 'blue'  # red = cause, blue = effect
        label = f"C{i+1}"
        plt.scatter(x, y, color=color)
        plt.text(x + 0.02, y, label, fontsize=10, fontweight='bold', color='black')

    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("D + R (Prominence)", fontsize=14)
    plt.ylabel("D - R (Relation)", fontsize=14)
    plt.title("DEMATEL Cause-Effect Diagram", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"📊 Grafik kaydedildi: {filename}")

# --- STEP X: Export Enhanced Results ---
def export_results(filename, crisp, normalized, total, D, R, prominence, relation,
                   svnn_opinions, ivnn_opinions, aggregated_ivnn):
    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(crisp).to_excel(writer, sheet_name="Crisp Matrix", index=False)
        pd.DataFrame(normalized).to_excel(writer, sheet_name="Normalized Matrix", index=False)
        pd.DataFrame(total).to_excel(writer, sheet_name="Total Relation Matrix", index=False)

        summary = pd.DataFrame({
            "D (Dispatching Power)": D,
            "R (Receiving Power)": R,
            "D + R (Prominence)": prominence,
            "D - R (Relation)": relation
        })
        summary.to_excel(writer, sheet_name="D-R Analysis", index_label="Factor")

        # SVNN T, I, F triple in one cell
        expert_count, f, _, _ = svnn_opinions.shape
        rows = []
        for e in range(expert_count):
            for i in range(f):
                for j in range(f):
                    T, I, F = svnn_opinions[e, i, j]
                    rows.append([f"Expert {e+1}", f"C{i+1}", f"C{j+1}", f"<{T:.2f}, {I:.2f}, {F:.2f}>"])
        pd.DataFrame(rows, columns=["Expert", "From", "To", "SVNN"]).to_excel(writer, sheet_name="SVNN Opinions", index=False)

        # IVNN Transformed Expert Opinions
        rows = []
        for e in range(expert_count):
            for i in range(f):
                for j in range(f):
                    T_L, T_U = ivnn_opinions[e, i, j, 0]
                    I_L, I_U = ivnn_opinions[e, i, j, 1]
                    F_L, F_U = ivnn_opinions[e, i, j, 2]
                    ivn_text = f"<[{T_L:.2f}, {T_U:.2f}], [{I_L:.2f}, {I_U:.2f}], [{F_L:.2f}, {F_U:.2f}]>"
                    rows.append([f"Expert {e+1}", f"C{i+1}", f"C{j+1}", ivn_text])
        pd.DataFrame(rows, columns=["Expert", "From", "To", "IVNN (Transformed)"]).to_excel(writer, sheet_name="IVNN Transformed", index=False)

        # IVNN Aggregated (as matrix f x f)
        matrix_data = []
        for i in range(f):
            row = []
            for j in range(f):
                T_L, T_U = aggregated_ivnn[i, j, 0]
                I_L, I_U = aggregated_ivnn[i, j, 1]
                F_L, F_U = aggregated_ivnn[i, j, 2]
                text = f"<[{T_L:.2f},{T_U:.2f}],[{I_L:.2f},{I_U:.2f}],[{F_L:.2f},{F_U:.2f}]>"
                row.append(text)
            matrix_data.append(row)
        df_agg_matrix = pd.DataFrame(matrix_data, columns=[f"C{j + 1}" for j in range(f)],
                                     index=[f"C{i + 1}" for i in range(f)])
        df_agg_matrix.to_excel(writer, sheet_name="IVNN Aggregated", index=True)


# --- MAIN EXECUTION ---
expert_opinions, expert_count, factor_count = read_excel_file()
svnn_opinions = convert_to_svnn(expert_opinions, linguistic_to_svnn)
ivnn_opinions = svnn_to_ivnn(svnn_opinions)
aggregated_ivnn = apply_ivnnwa(ivnn_opinions)
crisp_matrix = deneutrosophication(aggregated_ivnn)
normalized_matrix, k = normalize_crisp_matrix(crisp_matrix)
total_relation_matrix = compute_total_relation_matrix(normalized_matrix)
D, R, prominence, relation = compute_prominence_and_relation(total_relation_matrix)

# --- OUTPUT ---
print("Crisp matrix (shape):", crisp_matrix.shape)
print("Min:", np.min(crisp_matrix), "Max:", np.max(crisp_matrix))
print("Normalization factor:", k)
print("Toplam Etki Matrisi T ", total_relation_matrix.shape)
print("D (Dispatching Power):", D)
print("R (Receiving Power):", R)
print("D + R:", prominence)
print("D - R:", relation)

# --- EXPORT ---
export_results("RIVN_DEMATEL_Results.xlsx", crisp_matrix, normalized_matrix, total_relation_matrix,
               D, R, prominence, relation, svnn_opinions, ivnn_opinions, aggregated_ivnn)
print("✅ Sonuçlar Excel dosyasına yazıldı: RIVN_DEMATEL_Results.xlsx")

# --- PLOT WITH COLORS ---
plot_dematel_cause_effect(D, R)

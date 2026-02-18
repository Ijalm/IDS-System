import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================================
# COLOR PALETTE (matches IDS dashboard)
# ==========================================
COLOR_BG       = "#0a0e27"
COLOR_CARD     = "#141b2d"
COLOR_ACCENT   = "#1e88e5"
COLOR_SUCCESS  = "#00e676"
COLOR_WARNING  = "#ffc107"
COLOR_DANGER   = "#f44336"
COLOR_TEXT     = "#e0e0e0"
COLOR_TEXT_DIM = "#9e9e9e"

MODEL_COLORS   = ["#1e88e5", "#00e676", "#ffc107", "#f44336"]

# ==========================================
# Load Dataset
# ==========================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
dataset_file = os.path.join(BASE_DIR, "KDDTrain+.txt")

if not os.path.exists(dataset_file):
    print(f"ERROR: KDDTrain+.txt not found in {BASE_DIR}")
    exit()

column_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

print("Loading dataset...")
df = pd.read_csv(dataset_file, names=column_names)
df.drop("difficulty", axis=1, inplace=True)

le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    df[col] = le.fit_transform(df[col])

attack_mapping = {
    "normal":   0,
    "neptune":  1, "smurf": 1, "back": 1, "teardrop": 1, "pod": 1, "land": 1,
    "satan":    2, "ipsweep": 2, "nmap": 2, "portsweep": 2, "mscan": 2, "saint": 2,
}
df['label']  = df['label'].map(attack_mapping).fillna(2)

labels_dict  = {0: "Normal", 1: "DoS", 2: "Probe"}
target_names = [labels_dict[i] for i in sorted(labels_dict.keys())]

X = df.drop("label", axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ==========================================
# Train Models
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5),
    "SVM":           SVC(),
}

results = {}
best_model, best_accuracy, best_name = None, 0, ""

print("\n========== MODEL TRAINING ==========")
for name, model in models.items():
    print(f"Training {name}...")
    if name == "SVM":
        print("  -> Using 10,000-row subset for SVM...")
        model.fit(X_train[:10000], y_train[:10000])
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names,
                                   zero_division=0, output_dict=True)

    results[name] = {"acc": acc, "cm": cm, "report": report, "model": model}
    print(f"  -> {name} Accuracy: {acc:.4f}\n")

    if acc > best_accuracy:
        best_accuracy, best_model, best_name = acc, model, name

joblib.dump(best_model, os.path.join(BASE_DIR, "ids_model.pkl"))
joblib.dump(scaler,     os.path.join(BASE_DIR, "scaler.pkl"))
print(f"Saved Best Model: {best_name} ({best_accuracy:.4f})")

# ==========================================
# GUI Helpers
# ==========================================
def make_card(parent, pady=0):
    f = tk.Frame(parent, bg=COLOR_CARD)
    f.pack(fill="both", expand=True, padx=15, pady=pady)
    return f

def section_label(parent, text):
    tk.Label(parent, text=text, font=("Segoe UI", 9, "bold"),
             fg=COLOR_TEXT_DIM, bg=COLOR_CARD).pack(anchor="w", padx=18, pady=(14, 4))
    tk.Frame(parent, bg=COLOR_TEXT_DIM, height=1).pack(fill="x", padx=18, pady=(0, 8))

# Custom colormap: dark navy → teal/cyan → bright green
CM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ids_cm", ["#0a0e27", "#1e88e5", "#00e676"]
)

# ==========================================
# GUI
# ==========================================
def create_gui():
    root = tk.Tk()
    root.title("Detectify - Model Comparison")
    root.geometry("1300x820")
    root.configure(bg=COLOR_BG)

    # ── Header ──────────────────────────────────────────────
    header = tk.Frame(root, bg=COLOR_CARD, height=55)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(header, text="Detectify - Model Comparison",
             font=("Segoe UI", 16, "bold"), fg=COLOR_SUCCESS, bg=COLOR_CARD
             ).pack(side="left", padx=25, pady=12)

    tk.Label(header,
             text=f"Best Model: {best_name}   |   Accuracy: {best_accuracy:.4f}",
             font=("Segoe UI", 11), fg=COLOR_TEXT_DIM, bg=COLOR_CARD
             ).pack(side="right", padx=25, pady=12)

    # ── Notebook style ──────────────────────────────────────
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TNotebook",
                    background=COLOR_BG, borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab",
                    background=COLOR_CARD,
                    foreground=COLOR_TEXT_DIM,
                    padding=[20, 9],          # same padding always
                    font=("Segoe UI", 10),
                    borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", COLOR_ACCENT),
                          ("active",   "#1a2540")],
              foreground=[("selected", "white"),
                          ("active",   COLOR_TEXT)],
              # padding stays the same on selection → no shrink
              padding=[("selected", [20, 9])])

    style.configure("TFrame", background=COLOR_BG)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=0, pady=(8, 0))

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — Overview
    # ══════════════════════════════════════════════════════════
    overview_tab = tk.Frame(notebook, bg=COLOR_BG)
    notebook.add(overview_tab, text="   Overview   ")

    cards_row = tk.Frame(overview_tab, bg=COLOR_BG)
    cards_row.pack(fill="x", padx=15, pady=(18, 10))

    for idx, (name, data) in enumerate(results.items()):
        card = tk.Frame(cards_row, bg=COLOR_CARD)
        card.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(card, text=name, font=("Segoe UI", 10, "bold"),
                 fg=COLOR_TEXT_DIM, bg=COLOR_CARD).pack(anchor="w", padx=16, pady=(12, 2))
        col = COLOR_SUCCESS if data["acc"] == best_accuracy else COLOR_TEXT
        tk.Label(card, text=f"{data['acc']*100:.2f}%",
                 font=("Segoe UI", 22, "bold"), fg=col, bg=COLOR_CARD
                 ).pack(anchor="w", padx=16, pady=(0, 12))

    chart_card = tk.Frame(overview_tab, bg=COLOR_CARD)
    chart_card.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    tk.Label(chart_card, text="Accuracy Comparison",
             font=("Segoe UI", 12, "bold"), fg=COLOR_TEXT, bg=COLOR_CARD, anchor="w"
             ).pack(fill="x", padx=15, pady=(12, 4))

    fig_bar = Figure(figsize=(8, 4), dpi=100, facecolor=COLOR_CARD)
    ax_bar  = fig_bar.add_subplot(111)
    ax_bar.set_facecolor(COLOR_CARD)

    names_list = list(results.keys())
    accs       = [results[n]["acc"] for n in names_list]
    bar_objs   = ax_bar.bar(names_list, accs, color=MODEL_COLORS, width=0.5, edgecolor="none")

    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Accuracy", color=COLOR_TEXT_DIM, fontsize=10)
    ax_bar.tick_params(colors=COLOR_TEXT_DIM, labelsize=10)
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color(COLOR_TEXT_DIM)
    ax_bar.spines["bottom"].set_color(COLOR_TEXT_DIM)

    for bar in bar_objs:
        h = bar.get_height()
        ax_bar.annotate(f"{h:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", color=COLOR_TEXT, fontsize=9)

    fig_bar.tight_layout()
    FigureCanvasTkAgg(fig_bar, master=chart_card).get_tk_widget().pack(
        fill="both", expand=True, padx=10, pady=(0, 10))

    # ══════════════════════════════════════════════════════════
    #  TAB 2+ — Per-model tabs
    # ══════════════════════════════════════════════════════════
    for idx, (name, data) in enumerate(results.items()):
        tab = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(tab, text=f"   {name}   ")

        model_color = MODEL_COLORS[idx]
        report      = data["report"]   # dict form

        # ── 50 / 50 split using grid ──────────────────────────
        tab.columnconfigure(0, weight=1, uniform="half")
        tab.columnconfigure(1, weight=1, uniform="half")
        tab.rowconfigure(0, weight=1)

        # ── LEFT CARD ────────────────────────────────────────
        left_card = tk.Frame(tab, bg=COLOR_CARD)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(15, 6), pady=15)

        # Title + accuracy
        tk.Label(left_card, text=name,
                 font=("Segoe UI", 14, "bold"),
                 fg=model_color, bg=COLOR_CARD).pack(anchor="w", padx=18, pady=(18, 2))

        acc_col = COLOR_SUCCESS if data["acc"] == best_accuracy else COLOR_TEXT
        tk.Label(left_card, text=f"Accuracy: {data['acc']*100:.2f}%",
                 font=("Segoe UI", 11), fg=acc_col, bg=COLOR_CARD
                 ).pack(anchor="w", padx=18, pady=(0, 12))

        # ── Classification Report (modern table) ─────────────
        section_label(left_card, "CLASSIFICATION REPORT")

        tbl_frame = tk.Frame(left_card, bg=COLOR_CARD)
        tbl_frame.pack(fill="x", padx=18, pady=(0, 10))

        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        col_widths = [10, 10, 8, 10, 9]

        # Header row
        hdr_row = tk.Frame(tbl_frame, bg="#1e293b")
        hdr_row.pack(fill="x", pady=(0, 2))
        for h, w in zip(headers, col_widths):
            tk.Label(hdr_row, text=h, font=("Segoe UI", 9, "bold"),
                     fg=COLOR_TEXT_DIM, bg="#1e293b",
                     width=w, anchor="e" if h != "Class" else "w"
                     ).pack(side="left", padx=6, pady=5)

        # Class rows
        row_classes = target_names
        for i, cls in enumerate(row_classes):
            row_bg = COLOR_CARD if i % 2 == 0 else "#1a2235"
            r = tk.Frame(tbl_frame, bg=row_bg)
            r.pack(fill="x")

            cls_data = report.get(cls, {})
            vals = [
                cls,
                f"{cls_data.get('precision', 0):.2f}",
                f"{cls_data.get('recall',    0):.2f}",
                f"{cls_data.get('f1-score',  0):.2f}",
                f"{int(cls_data.get('support', 0))}",
            ]
            # Class name colored
            tk.Label(r, text=vals[0], font=("Segoe UI", 9, "bold"),
                     fg=model_color, bg=row_bg, width=col_widths[0], anchor="w"
                     ).pack(side="left", padx=6, pady=4)
            for v, w in zip(vals[1:], col_widths[1:]):
                tk.Label(r, text=v, font=("Consolas", 9),
                         fg=COLOR_TEXT, bg=row_bg, width=w, anchor="e"
                         ).pack(side="left", padx=6, pady=4)

        # Separator + accuracy / avg rows
        tk.Frame(tbl_frame, bg=COLOR_TEXT_DIM, height=1).pack(fill="x", pady=(4, 0))

        avg_rows = [
            ("Accuracy", "", "", f"{report.get('accuracy', 0):.2f}",
             f"{int(report.get('weighted avg', {}).get('support', 0))}"),
            ("Macro Avg",
             f"{report.get('macro avg', {}).get('precision', 0):.2f}",
             f"{report.get('macro avg', {}).get('recall',    0):.2f}",
             f"{report.get('macro avg', {}).get('f1-score',  0):.2f}",
             f"{int(report.get('macro avg', {}).get('support', 0))}"),
            ("Weighted Avg",
             f"{report.get('weighted avg', {}).get('precision', 0):.2f}",
             f"{report.get('weighted avg', {}).get('recall',    0):.2f}",
             f"{report.get('weighted avg', {}).get('f1-score',  0):.2f}",
             f"{int(report.get('weighted avg', {}).get('support', 0))}"),
        ]
        for vals in avg_rows:
            r = tk.Frame(tbl_frame, bg=COLOR_CARD)
            r.pack(fill="x")
            tk.Label(r, text=vals[0], font=("Segoe UI", 9, "italic"),
                     fg=COLOR_TEXT_DIM, bg=COLOR_CARD, width=col_widths[0], anchor="w"
                     ).pack(side="left", padx=6, pady=3)
            for v, w in zip(vals[1:], col_widths[1:]):
                tk.Label(r, text=v, font=("Consolas", 9),
                         fg=COLOR_TEXT_DIM, bg=COLOR_CARD, width=w, anchor="e"
                         ).pack(side="left", padx=6, pady=3)

        # ── RIGHT CARD — Confusion Matrix ─────────────────────
        right_card = tk.Frame(tab, bg=COLOR_CARD)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(6, 15), pady=15)

        tk.Label(right_card, text="Confusion Matrix",
                 font=("Segoe UI", 12, "bold"), fg=COLOR_TEXT, bg=COLOR_CARD, anchor="w"
                 ).pack(fill="x", padx=18, pady=(18, 6))

        fig_cm = Figure(figsize=(5, 4.5), dpi=100, facecolor=COLOR_CARD)
        ax_cm  = fig_cm.add_subplot(111)
        ax_cm.set_facecolor(COLOR_CARD)

        sns.heatmap(data["cm"], annot=True, fmt="d",
                    cmap=CM_CMAP,
                    xticklabels=target_names,
                    yticklabels=target_names,
                    ax=ax_cm,
                    linewidths=0.8, linecolor=COLOR_BG,
                    annot_kws={"size": 12, "weight": "bold", "color": COLOR_TEXT})

        ax_cm.set_xlabel("Classified As", color=COLOR_TEXT_DIM, fontsize=10, labelpad=10)
        ax_cm.set_ylabel("Actual",        color=COLOR_TEXT_DIM, fontsize=10, labelpad=10)
        ax_cm.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)

        # Colour bar text to match palette
        cbar = ax_cm.collections[0].colorbar
        cbar.ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=8)
        cbar.outline.set_edgecolor(COLOR_TEXT_DIM)

        fig_cm.tight_layout(pad=2)
        FigureCanvasTkAgg(fig_cm, master=right_card).get_tk_widget().pack(
            fill="both", expand=True, padx=10, pady=(0, 15))

    root.mainloop()

create_gui()
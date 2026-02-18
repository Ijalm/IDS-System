import tkinter as tk
from tkinter import ttk
import pandas as pd, numpy as np, joblib, threading, time, os
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder

# =========================
# PATHS & GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(f"{BASE_DIR}/ids_model.pkl")
scaler   = joblib.load(f"{BASE_DIR}/scaler.pkl")
log_file = f"{BASE_DIR}/log.txt"

# =========================
# COLOR PALETTE
# =========================
COLORS = {"Normal": "#00e676", "DoS": "#f44336", "Probe": "#ffc107"}
BG     = "#0a0e27"
CARD   = "#141b2d"
ACCENT = "#1e88e5"
TEXT   = "#e0e0e0"
DIM    = "#9e9e9e"

BTN_START_ACTIVE   = "#00e676"
BTN_START_DISABLED = "#1a4a2e"
BTN_STOP_ACTIVE    = "#f44336"
BTN_STOP_DISABLED  = "#4a1a1a"
BTN_FG_ACTIVE      = "white"
BTN_FG_DISABLED    = "#555555"

# =========================
# STATE
# =========================
state   = {"running": False, "packets": 0, "prev_pkt": None}
alerts  = {"Normal": 0, "DoS": 0, "Probe": 0}
metrics = {}

# =========================
# DATA PREPARATION
# =========================
cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

df = pd.read_csv(f"{BASE_DIR}/KDDTest+.txt", names=cols).drop("difficulty", axis=1)
for col in ['protocol_type', 'service', 'flag']:
    df[col] = LabelEncoder().fit_transform(df[col])

atk_map = {
    "normal":0, "neptune":1, "smurf":1, "back":1, "teardrop":1,
    "pod":1, "land":1, "satan":2, "ipsweep":2, "nmap":2, "portsweep":2
}
df['label'] = df['label'].map(atk_map).fillna(2)
X_df = df.drop("label", axis=1)
X    = scaler.transform(X_df)

# =========================
# BUTTON HELPERS
# =========================
def set_buttons(start_state, stop_state):
    start_btn.config(
        state=start_state,
        bg=BTN_START_ACTIVE    if start_state == "normal" else BTN_START_DISABLED,
        fg=BTN_FG_ACTIVE       if start_state == "normal" else BTN_FG_DISABLED,
        cursor="hand2"         if start_state == "normal" else "arrow",
        activebackground="#00c853" if start_state == "normal" else BTN_START_DISABLED
    )
    stop_btn.config(
        state=stop_state,
        bg=BTN_STOP_ACTIVE     if stop_state == "normal" else BTN_STOP_DISABLED,
        fg=BTN_FG_ACTIVE       if stop_state == "normal" else BTN_FG_DISABLED,
        cursor="hand2"         if stop_state == "normal" else "arrow",
        activebackground="#d32f2f" if stop_state == "normal" else BTN_STOP_DISABLED
    )

# =========================
# LOGGING
# =========================
def log_event(msg, to_file=True):
    if to_file:
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

# =========================
# ALERT LOG ROW INSERT
# =========================
def insert_alert_row(ts, packet_num, attack_type):
    color = COLORS.get(attack_type, TEXT)
    log_text.config(state="normal")

    # alternating row bg via tag trick — use a unique tag per row
    tag = f"row_{packet_num}"
    row_bg = "#0d1117" if (log_text.count("1.0", tk.END, "lines")[0] % 2 == 0) else "#111827"

    log_text.insert(tk.END, f"  {ts:<12}", (tag,))
    log_text.insert(tk.END, f"#{str(packet_num):<8}", (tag,))
    log_text.insert(tk.END, f"{attack_type:<10}", (f"type_{tag}",))
    log_text.insert(tk.END, f"Packet #{packet_num}\n", (tag,))

    log_text.tag_config(tag,        foreground=DIM,   background=row_bg)
    log_text.tag_config(f"type_{tag}", foreground=color, background=row_bg)
    log_text.see(tk.END)
    log_text.config(state="disabled")

# =========================
# FEATURE ROW INSERT
# =========================
def update_feature_panel(attack_type, changed_features):
    color = COLORS.get(attack_type, TEXT)
    feature_text.config(state="normal")
    feature_text.delete("1.0", tk.END)

    # Header row
    feature_text.insert(tk.END, f"  {'#':<6}{'Feature':<30}{'Status'}\n", "header")
    feature_text.insert(tk.END, f"  {'─'*48}\n", "dim")

    for i, feat in enumerate(changed_features):
        row_bg  = "#0d1117" if i % 2 == 0 else "#111827"
        tag_row = f"feat_{i}"
        tag_val = f"val_{i}"
        feature_text.insert(tk.END, f"  {str(i+1):<6}", (tag_row,))
        feature_text.insert(tk.END, f"{feat:<30}", (tag_row,))
        feature_text.insert(tk.END, f"CHANGED\n",   (tag_val,))
        feature_text.tag_config(tag_row, foreground=DIM,   background=row_bg)
        feature_text.tag_config(tag_val, foreground=color, background=row_bg)

    feature_text.tag_config("header", foreground=TEXT,    background=CARD, font=("Consolas", 9, "bold"))
    feature_text.tag_config("dim",    foreground="#2a3a4a",background=CARD)
    feature_text.config(state="disabled")

# =========================
# CORE LOGIC
# =========================
def run_demo():
    for row in X:
        if not state["running"]:
            break

        attack = {0:"Normal", 1:"DoS", 2:"Probe"}.get(
            model.predict(row.reshape(1,-1))[0], "Probe"
        )
        alerts[attack]   += 1
        state["packets"] += 1

        # Feature analysis
        if state["prev_pkt"] is not None and attack in ["DoS", "Probe"]:
            diff_idx = np.where(np.abs(state["prev_pkt"] - row) > 0.01)[0]
            changed  = [X_df.columns[i] for i in diff_idx[:10]] if len(diff_idx) > 0 else ["No significant changes"]
            root.after(0, lambda c=changed, a=attack: update_feature_panel(a, c))

        state["prev_pkt"] = row.copy()

        if attack != "Normal":
            ts  = datetime.now().strftime("%H:%M:%S")
            msg = f"[{ts}] {attack.upper()} ATTACK DETECTED - Packet #{state['packets']}"
            log_event(msg)
            root.after(0, lambda t=ts, p=state["packets"], a=attack: insert_alert_row(t, p, a))

        root.after(0, update_gui)
        time.sleep(0.08)

def update_gui():
    for atk, count in alerts.items():
        metrics[atk]["val"].config(text=str(count))
        if state["packets"] > 0:
            metrics[atk]["pct"].config(text=f"{(count/state['packets'])*100:.1f}% of traffic")

    # Total packets card
    total_val_label.config(text=str(state["packets"]))

    for i, bar in enumerate(bars):
        bar.set_height(list(alerts.values())[i])
    ax_bar.set_ylim(0, max(max(alerts.values()) * 1.1, 10))
    canvas_bar.draw_idle()

def toggle_monitor(start=True):
    state["running"] = start
    if start:
        with open(log_file, 'w') as f:
            f.write(f"{'='*60}\nIDS LOG STARTED: {datetime.now()}\n{'='*60}\n\n")
            f.write(f"{'Timestamp':<12} {'Type':<10} {'Detail'}\n")
            f.write(f"{'-'*60}\n")
        set_buttons("disabled", "normal")
        threading.Thread(target=run_demo, daemon=True).start()
    else:
        log_event(f"\nSession Stopped. Total Packets: {state['packets']}\n{'='*40}")
        set_buttons("normal", "disabled")

# =========================
# GUI SETUP
# =========================
root = tk.Tk()
root.title("Detectify - Classification of Test Dataset")
root.geometry("1400x850")
root.minsize(1200, 700)
root.configure(bg=BG)

# ── Metric Cards Row ─────────────────────────────────────
metrics_frame = tk.Frame(root, bg=BG)
metrics_frame.pack(fill="x", padx=20, pady=(15, 5))

# Total Packets card (styled like attack cards)
total_card = tk.Frame(metrics_frame, bg=CARD)
total_card.pack(side="left", fill="both", expand=True, padx=5)
tk.Label(total_card, text="Total Packets",
         font=("Segoe UI", 14, "bold"),
         fg=ACCENT, bg=CARD).pack(anchor="w", padx=15, pady=(10, 0))
total_val_label = tk.Label(total_card, text="0",
                            font=("Segoe UI", 26, "bold"),
                            fg=ACCENT, bg=CARD)
total_val_label.pack(anchor="w", padx=15, pady=(0, 10))

# Normal / DoS / Probe cards
for atk, color in COLORS.items():
    card = tk.Frame(metrics_frame, bg=CARD)
    card.pack(side="left", fill="both", expand=True, padx=5)
    tk.Label(card, text=atk,
             font=("Segoe UI", 14, "bold"),
             fg=color, bg=CARD).pack(anchor="w", padx=15, pady=(10, 0))
    val_lbl = tk.Label(card, text="0",
                       font=("Segoe UI", 26, "bold"),
                       fg=color, bg=CARD)
    val_lbl.pack(anchor="w", padx=15)
    pct_lbl = tk.Label(card, text="",
                       font=("Segoe UI", 8),
                       fg=DIM, bg=CARD)
    pct_lbl.pack(anchor="w", padx=15, pady=(0, 10))
    metrics[atk] = {"val": val_lbl, "pct": pct_lbl}

# ── Main Content ─────────────────────────────────────────
content = tk.Frame(root, bg=BG)
content.pack(fill="both", expand=True, padx=20, pady=5)

# Left: Bar Chart
left_panel = tk.Frame(content, bg=CARD)
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

tk.Label(left_panel, text="ATTACK DISTRIBUTION",
         font=("Segoe UI", 11, "bold"),
         fg=TEXT, bg=CARD, anchor="w").pack(fill="x", padx=12, pady=(10, 3))

fig_bar = Figure(figsize=(6, 5), facecolor=CARD, dpi=100)
ax_bar  = fig_bar.add_subplot(111, facecolor=CARD)
bars    = ax_bar.bar(COLORS.keys(), alerts.values(),
                     color=COLORS.values(), width=0.6, edgecolor="none")

ax_bar.set_ylim(0, 10)
ax_bar.set_ylabel("Packet Count",          color=DIM, fontsize=10)
ax_bar.set_xlabel("Attack Classification", color=DIM, fontsize=10)
ax_bar.tick_params(colors=DIM, labelsize=9)
for spine in ['top', 'right']:
    ax_bar.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax_bar.spines[spine].set_color(DIM)
fig_bar.tight_layout()

canvas_bar = FigureCanvasTkAgg(fig_bar, master=left_panel)
canvas_bar.draw()
canvas_bar.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

# Right: Log + Feature panels
right_panel = tk.Frame(content, bg=BG, width=380)
right_panel.pack(side="right", fill="both")
right_panel.pack_propagate(False)

# Force exact 50/50 vertical split using grid and uniform weights
right_panel.rowconfigure(0, weight=1, uniform="half")
right_panel.rowconfigure(1, weight=1, uniform="half")
right_panel.columnconfigure(0, weight=1)

def create_text_card(parent, title, row, fg_color=TEXT):
    f = tk.Frame(parent, bg=CARD)
    # Changed from pack to grid to respect the row weights defined above
    f.grid(row=row, column=0, sticky="nsew", pady=3)
    
    tk.Label(f, text=title,
             font=("Segoe UI", 11, "bold"),
             fg=TEXT, bg=CARD, anchor="w").pack(fill="x", padx=12, pady=(10, 3))
    if title=="SECURITY ALERTS":
    
      # Column header row
        hdr_frame = tk.Frame(f, bg="#1e293b")
        hdr_frame.pack(fill="x", padx=12, pady=(0, 2))
        for col_txt, col_w in [("Timestamp", 12), ("Pkt ID", 8), ("Type", 10), ("Detail", 14)]:
            tk.Label(hdr_frame, text=col_txt,
                font=("Segoe UI", 8, "bold"),
                fg=DIM, bg="#1e293b",
                width=col_w, anchor="w").pack(side="left", padx=4, pady=4)       
    txt = tk.Text(f, bg="#0d1117", fg=fg_color,
                    font=("Consolas", 9),
                    relief="flat", wrap="word",
                    height=1, state="disabled")
    txt.pack(fill="both", expand=True, padx=12, pady=(0, 10))
    return txt

# Pass the specific grid row (0 or 1) into the helper function
log_text     = create_text_card(right_panel, "SECURITY ALERTS", row=0, fg_color="#f44336")
feature_text = create_text_card(right_panel, "FEATURE ANALYSIS", row=1, fg_color="#00e676")

# ── Control Panel ─────────────────────────────────────────
ctrl_frame = tk.Frame(root, bg=CARD, height=60)
ctrl_frame.pack(fill="x", padx=20, pady=(5, 15))
ctrl_frame.pack_propagate(False)

btn_inner = tk.Frame(ctrl_frame, bg=CARD)
btn_inner.pack(expand=True)

start_btn = tk.Button(btn_inner,
                      text="START MONITORING",
                      font=("Segoe UI", 10, "bold"),
                      bg=BTN_START_ACTIVE, fg=BTN_FG_ACTIVE,
                      activebackground="#00c853",
                      relief="flat", cursor="hand2",
                      padx=30, pady=10,
                      command=lambda: toggle_monitor(True))
start_btn.pack(side="left", padx=8)

stop_btn = tk.Button(btn_inner,
                     text="STOP MONITORING",
                     font=("Segoe UI", 10, "bold"),
                     bg=BTN_STOP_DISABLED, fg=BTN_FG_DISABLED,
                     activebackground="#d32f2f",
                     relief="flat", cursor="arrow",
                     padx=30, pady=10,
                     state="disabled",
                     command=lambda: toggle_monitor(False))
stop_btn.pack(side="left", padx=8)

# =========================
# RUN
# =========================
root.mainloop()
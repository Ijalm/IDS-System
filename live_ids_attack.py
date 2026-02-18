# ==============================
# LIVE IDS — Modern GUI
# ==============================

from scapy.all import sniff, IP, TCP, get_if_list
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# PATHS
# =========================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
model         = joblib.load(os.path.join(BASE_DIR, "ids_model.pkl"))
scaler        = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
LOG_FILE      = os.path.join(BASE_DIR, "live_ids_log.txt")

# =========================
# COLOR PALETTE
# =========================
COLOR_BG       = "#0a0e27"
COLOR_CARD     = "#141b2d"
COLOR_ACCENT   = "#1e88e5"
COLOR_SUCCESS  = "#00e676"
COLOR_WARNING  = "#ffc107"
COLOR_DANGER   = "#f44336"
COLOR_TEXT     = "#e0e0e0"
COLOR_TEXT_DIM = "#9e9e9e"

ATTACK_COLORS  = {
    "Normal": COLOR_SUCCESS,
    "DoS":    COLOR_DANGER,
    "Probe":  COLOR_WARNING,
}

# =========================
# IDS STATE
# =========================
labels_dict    = {0: "Normal", 1: "DoS", 2: "Probe"}
alert_count    = {"Normal": 0, "DoS": 0, "Probe": 0}
total_packets  = 0
packet_id      = 0
running        = False
session_start  = None

TIME_WINDOW         = 3
connection_history  = []
syn_counter         = defaultdict(int)
port_scan_tracker   = defaultdict(set)

# =========================
# FEATURE COLUMNS
# =========================
feature_columns = [
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
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# =========================
# LOGGING
# =========================
def init_log():
    with open(LOG_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LIVE INTRUSION DETECTION SYSTEM — LOG FILE\n")
        f.write(f"Session Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Timestamp':<12} {'Pkt ID':<10} {'Type':<10} {'Source IP':<18} {'Detail'}\n")
        f.write("-" * 80 + "\n")

def log_entry(ts, pkt_id, attack_type, src_ip, detail=""):
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts:<12} #{pkt_id:<9} {attack_type:<10} {src_ip:<18} {detail}\n")

def close_log():
    with open(LOG_FILE, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Session Ended   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Packets   : {total_packets}\n")
        for k, v in alert_count.items():
            f.write(f"  {k:<12}: {v}\n")
        f.write("=" * 80 + "\n")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(packet):
    try:
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            return None, None

        current_time = time.time()
        src_ip       = packet[IP].src
        tcp          = packet[TCP]
        features     = np.zeros(41)

        protocol_map  = {6: 0, 17: 1, 1: 2}
        features[1]   = protocol_map.get(packet[IP].proto, 0)
        features[2]   = tcp.dport
        tcp_flag      = str(tcp.flags)
        features[3]   = 0 if tcp_flag == "S" else 1
        features[4]   = len(packet)

        connection_history.append((current_time, src_ip))
        while connection_history and current_time - connection_history[0][0] > TIME_WINDOW:
            connection_history.pop(0)

        features[22] = sum(1 for t, ip in connection_history if ip == src_ip)

        if tcp_flag == "S":
            syn_counter[src_ip] += 1
        features[24] = syn_counter[src_ip]

        port_scan_tracker[src_ip].add(tcp.dport)

        return features.reshape(1, -1), src_ip

    except Exception as e:
        return None, None

# =========================
# DETECTION LOGIC
# =========================
def detect_packet(packet):
    global total_packets, packet_id

    features, src_ip = extract_features(packet)
    if features is None:
        return

    total_packets += 1
    packet_id     += 1

    features_df     = pd.DataFrame(features, columns=feature_columns)
    features_scaled = scaler.transform(features_df)
    prediction      = model.predict(features_scaled)
    attack_type     = labels_dict.get(int(prediction[0]), "Normal")

    # Override rules
    if syn_counter[src_ip] > 50 and len(port_scan_tracker[src_ip]) < 10:
        attack_type = "DoS"
    elif len(port_scan_tracker[src_ip]) > 20:
        attack_type = "Probe"

    if attack_type not in alert_count:
        attack_type = "Normal"
    alert_count[attack_type] += 1

    # Reset if thresholds exceeded
    if syn_counter[src_ip] > 300:
        syn_counter[src_ip]        = 0
        port_scan_tracker[src_ip]  = set()

    ts     = datetime.now().strftime("%H:%M:%S")
    detail = (f"SYN={syn_counter[src_ip]} "
              f"Ports={len(port_scan_tracker[src_ip])}")

    log_entry(ts, packet_id, attack_type, src_ip, detail)

    # Push update to GUI (thread-safe)
    root.after(0, lambda: update_gui(ts, packet_id, attack_type, src_ip, detail))

# =========================
# SNIFF THREAD
# =========================
def sniff_thread():
    sniff(iface=iface_var.get(), filter="tcp",
          prn=detect_packet, store=False,
          stop_filter=lambda p: not running)

# =========================
# GUI FUNCTIONS
# =========================
def set_buttons(start_state, stop_state):
    """Enable or disable buttons and update their appearance."""
    start_btn.config(
        state=start_state,
        bg=COLOR_SUCCESS if start_state == "normal" else "#4a6b4a",
        fg="white" if start_state == "normal" else "#7a9a7a",
        cursor="hand2" if start_state == "normal" else "arrow"
    )
    stop_btn.config(
        state=stop_state,
        bg=COLOR_DANGER if stop_state == "normal" else "#6b3a3a",
        fg="white" if stop_state == "normal" else "#9a6a6a",
        cursor="hand2" if stop_state == "normal" else "arrow"
    )

def start_capture():
    global running, session_start
    if running:
        return
    running        = True
    session_start  = datetime.now()
    init_log()
    set_buttons("disabled", "normal")
    threading.Thread(target=sniff_thread, daemon=True).start()

def stop_capture():
    global running
    running = False
    close_log()
    set_buttons("normal", "disabled")

def update_gui(ts, pkt_id, attack_type, src_ip, detail):
    # Update counters
    for k in alert_count:
        counter_labels[k].config(text=str(alert_count[k]))
        pct = alert_count[k] / max(total_packets, 1) * 100
        pct_labels[k].config(text=f"{pct:.1f}% of traffic")

    # Total packets
    total_label.config(text=str(total_packets))

    # Update bar chart
    for i, k in enumerate(alert_count):
        bars[i].set_height(alert_count[k])
    max_v = max(alert_count.values()) if max(alert_count.values()) > 0 else 10
    ax_bar.set_ylim(0, max_v * 1.15)
    canvas_bar.draw_idle()

    # Log to UI (attacks only)
    if attack_type != "Normal":
        color = ATTACK_COLORS.get(attack_type, COLOR_TEXT)
        log_text.config(state="normal")
        log_text.insert(
            tk.END,
            f"[{ts}]  #{pkt_id:<6}  {attack_type:<8}  {src_ip}\n",
            attack_type
        )
        log_text.see(tk.END)
        log_text.config(state="disabled")

# =========================
# GUI BUILD
# =========================
root = tk.Tk()
root.title("Detectify - Live Attack Classification")
root.geometry("1400x850")
root.minsize(1200, 700)
root.configure(bg=COLOR_BG)

iface_var = tk.StringVar(value="Wi-Fi")

# ── Metric Cards ──────────────────────────────────────────
cards_frame = tk.Frame(root, bg=COLOR_BG)
cards_frame.pack(fill="x", padx=15, pady=(15, 5))

counter_labels = {}
pct_labels     = {}

# Total packets card
total_card = tk.Frame(cards_frame, bg=COLOR_CARD)
total_card.pack(side="left", fill="both", expand=True, padx=5)
tk.Label(total_card, text="Total Packets",
         font=("Segoe UI", 10, "bold"),
         fg=COLOR_TEXT_DIM, bg=COLOR_CARD).pack(anchor="w", padx=16, pady=(12, 2))
total_label = tk.Label(total_card, text="0",
                       font=("Segoe UI", 26, "bold"),
                       fg=COLOR_ACCENT, bg=COLOR_CARD)
total_label.pack(anchor="w", padx=16, pady=(0, 12))

# Attack type cards
for attack_type, color in ATTACK_COLORS.items():
    card = tk.Frame(cards_frame, bg=COLOR_CARD)
    card.pack(side="left", fill="both", expand=True, padx=5)

    tk.Label(card, text=attack_type,
             font=("Segoe UI", 12, "bold"),
             fg=color, bg=COLOR_CARD).pack(anchor="w", padx=16, pady=(12, 2))

    val_lbl = tk.Label(card, text="0",
                       font=("Segoe UI", 26, "bold"),
                       fg=color, bg=COLOR_CARD)
    val_lbl.pack(anchor="w", padx=16, pady=(0, 2))

    pct_lbl = tk.Label(card, text="0.0% of traffic",
                       font=("Segoe UI", 8),
                       fg=COLOR_TEXT_DIM, bg=COLOR_CARD)
    pct_lbl.pack(anchor="w", padx=16, pady=(0, 10))

    counter_labels[attack_type] = val_lbl
    pct_labels[attack_type]     = pct_lbl

# ── Main content area ─────────────────────────────────────
content = tk.Frame(root, bg=COLOR_BG)
content.pack(fill="both", expand=True, padx=15, pady=5)

# Left — bar chart
left_panel = tk.Frame(content, bg=COLOR_CARD)
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

tk.Label(left_panel, text="ATTACK DISTRIBUTION",
         font=("Segoe UI", 11, "bold"),
         fg=COLOR_TEXT, bg=COLOR_CARD, anchor="w"
         ).pack(fill="x", padx=12, pady=(12, 4))

fig_bar = Figure(figsize=(5, 4), facecolor=COLOR_CARD, dpi=100)
ax_bar  = fig_bar.add_subplot(111)
ax_bar.set_facecolor(COLOR_CARD)

bar_colors = [ATTACK_COLORS[k] for k in alert_count]
bars       = ax_bar.bar(list(alert_count.keys()),
                        list(alert_count.values()),
                        color=bar_colors, width=0.55, edgecolor="none")

ax_bar.set_ylim(0, 10)
ax_bar.set_ylabel("Packet Count", color=COLOR_TEXT_DIM, fontsize=10)
ax_bar.set_xlabel("Classification", color=COLOR_TEXT_DIM, fontsize=10)
ax_bar.tick_params(colors=COLOR_TEXT_DIM, labelsize=10)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_color(COLOR_TEXT_DIM)
ax_bar.spines["bottom"].set_color(COLOR_TEXT_DIM)
fig_bar.tight_layout()

canvas_bar = FigureCanvasTkAgg(fig_bar, master=left_panel)
canvas_bar.draw()
canvas_bar.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 10))

# Right — log
right_panel = tk.Frame(content, bg=COLOR_BG, width=480)
right_panel.pack(side="right", fill="both", expand=False)
right_panel.pack_propagate(False)

log_card = tk.Frame(right_panel, bg=COLOR_CARD)
log_card.pack(fill="both", expand=True)

tk.Label(log_card, text="SECURITY ALERTS",
         font=("Segoe UI", 11, "bold"),
         fg=COLOR_TEXT, bg=COLOR_CARD, anchor="w"
         ).pack(fill="x", padx=12, pady=(12, 4))

# Column headers
hdr = tk.Frame(log_card, bg="#1e293b")
hdr.pack(fill="x", padx=12, pady=(0, 2))
for txt, w in [("Timestamp", 10), ("Pkt ID", 8), ("Type", 8), ("Source IP", 16)]:
    tk.Label(hdr, text=txt, font=("Segoe UI", 8, "bold"),
             fg=COLOR_TEXT_DIM, bg="#1e293b", width=w, anchor="w"
             ).pack(side="left", padx=6, pady=5)

log_text = tk.Text(log_card, bg="#0d1117", fg=COLOR_TEXT,
                   font=("Consolas", 9), relief="flat",
                   wrap="none", state="disabled")
log_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

# Tag colors per type
log_text.tag_config("DoS",   foreground=COLOR_DANGER)
log_text.tag_config("Probe", foreground=COLOR_WARNING)

# ── Control Panel ─────────────────────────────────────────
ctrl = tk.Frame(root, bg=COLOR_CARD, height=60)
ctrl.pack(fill="x", padx=0, pady=(5, 0))
ctrl.pack_propagate(False)

btn_frame = tk.Frame(ctrl, bg=COLOR_CARD)
btn_frame.pack(expand=True)

start_btn = tk.Button(btn_frame,
                      text="START MONITORING",
                      font=("Segoe UI", 10, "bold"),
                      bg=COLOR_SUCCESS, fg="white",
                      activebackground="#00c853",
                      relief="flat", cursor="hand2",
                      padx=30, pady=10,
                      command=start_capture)
start_btn.pack(side="left", padx=10)

stop_btn = tk.Button(btn_frame,
                     text="STOP MONITORING",
                     font=("Segoe UI", 10, "bold"),
                     bg="#6b3a3a", fg="#9a6a6a",
                     activebackground="#d32f2f",
                     relief="flat", cursor="arrow",
                     padx=30, pady=10,
                     state="disabled",
                     command=stop_capture)
stop_btn.pack(side="left", padx=10)

# =========================
# RUN
# =========================
print("Available Interfaces:", get_if_list())
root.mainloop()
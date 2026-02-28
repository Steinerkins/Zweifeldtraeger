import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Interaktives Statik-Tool", layout="wide")
st.title("Statik-Tool: Träger mit Kragarmen")

def berechne_traeger(L_k1, L1, L2, L_k2, q_k1, q1, q2, q_k2, F, x_F_rel):
    x_A = L_k1
    x_F = L_k1 + x_F_rel # Umrechnung relativ zu A in absolute Koordinate
    
    def macaulay(x_vals, a, power):
        if power == 0: return np.where(x_vals >= a, 1.0, 0.0)
        else: return np.where(x_vals > a, (x_vals - a)**power, 0.0)
            
    def get_load_M(x, q, a, b):
        return - (q / 2.0) * macaulay(x, a, 2) + (q / 2.0) * macaulay(x, b, 2)

    def get_load_V(x, q, a, b):
        return - q * macaulay(x, a, 1) + q * macaulay(x, b, 1)

    if L2 == 0:
        # --- Fall 1: Einfeldträger mit Kragarmen (statisch bestimmt) ---
        x_B = L_k1 + L1
        L_tot = L_k1 + L1 + L_k2
        x = np.linspace(0, L_tot, 5000)
        x_C = x_B # Dummy, Auflager C fällt weg
        
        F_tot = q_k1*L_k1 + q1*L1 + q_k2*L_k2 + F
        
        # Momentensumme um B zur Bestimmung von Auflager A
        M_B_q = (q_k1*L_k1*(x_B - L_k1/2) + 
                 q1*L1*(x_B - (L_k1 + L1/2)) + 
                 q_k2*L_k2*(x_B - (L_k1 + L1 + L_k2/2)))
        M_B_F = F * (x_B - x_F)
        
        A_y = (M_B_q + M_B_F) / (x_B - x_A)
        B_y = F_tot - A_y
        C_y = 0.0 # Gibt es nicht
        
        # Schnittgrößen
        M_load = (get_load_M(x, q_k1, 0, L_k1) + 
                  get_load_M(x, q1, L_k1, x_B) + 
                  get_load_M(x, q_k2, x_B, L_tot))
        M = A_y * macaulay(x, x_A, 1) + B_y * macaulay(x, x_B, 1) + M_load - F * macaulay(x, x_F, 1)
        
        V_load = (get_load_V(x, q_k1, 0, L_k1) + 
                  get_load_V(x, q1, L_k1, x_B) + 
                  get_load_V(x, q_k2, x_B, L_tot))
        V = A_y * macaulay(x, x_A, 0) + B_y * macaulay(x, x_B, 0) + V_load - F * macaulay(x, x_F, 0)
        
    else:
        # --- Fall 2: Zweifeldträger mit Kragarmen (statisch unbestimmt) ---
        x_B = L_k1 + L1
        x_C = L_k1 + L1 + L2
        L_tot = L_k1 + L1 + L2 + L_k2
        x = np.linspace(0, L_tot, 5000)
        
        F_tot = q_k1*L_k1 + q1*L1 + q2*L2 + q_k2*L_k2 + F
        
        # 0-System (Mittelstütze B entfernt)
        M_C_q = (q_k1*L_k1*(x_C - L_k1/2) + 
                 q1*L1*(x_C - (L_k1 + L1/2)) + 
                 q2*L2*(x_C - (L_k1 + L1 + L2/2)) + 
                 q_k2*L_k2*(x_C - (L_k1 + L1 + L2 + L_k2/2)))
        M_C_F = F * (x_C - x_F)
        
        A_y0 = (M_C_q + M_C_F) / (x_C - x_A)
        C_y0 = F_tot - A_y0
        
        M0_load = (get_load_M(x, q_k1, 0, L_k1) + get_load_M(x, q1, L_k1, x_B) + 
                   get_load_M(x, q2, x_B, x_C) + get_load_M(x, q_k2, x_C, L_tot))
        M0 = A_y0 * macaulay(x, x_A, 1) + C_y0 * macaulay(x, x_C, 1) + M0_load - F * macaulay(x, x_F, 1)
        
        # 1-System
        A_y1 = - (x_C - x_B) / (x_C - x_A)
        C_y1 = -1.0 - A_y1
        M1 = A_y1 * macaulay(x, x_A, 1) + 1.0 * macaulay(x, x_B, 1) + C_y1 * macaulay(x, x_C, 1)
        
        # Kraftgrößenverfahren
        dx = x[1] - x[0]
        delta_10 = np.sum(M0 * M1) * dx
        delta_11 = np.sum(M1 * M1) * dx
        B_y = - delta_10 / delta_11
        
        # Superposition
        A_y = A_y0 + B_y * A_y1
        C_y = C_y0 + B_y * C_y1
        M = M0 + B_y * M1
        
        V0_load = (get_load_V(x, q_k1, 0, L_k1) + get_load_V(x, q1, L_k1, x_B) + 
                   get_load_V(x, q2, x_B, x_C) + get_load_V(x, q_k2, x_C, L_tot))
        V0 = A_y0 * macaulay(x, x_A, 0) + C_y0 * macaulay(x, x_C, 0) + V0_load - F * macaulay(x, x_F, 0)
        V1 = A_y1 * macaulay(x, x_A, 0) + 1.0 * macaulay(x, x_B, 0) + C_y1 * macaulay(x, x_C, 0)
        V = V0 + B_y * V1

    # Max/Min Momente und deren Position relativ zu Auflager A
    max_M = np.max(M)
    min_M = np.min(M)
    x_max_M = x[np.argmax(M)] - x_A
    x_min_M = x[np.argmin(M)] - x_A

    return x, V, M, A_y, B_y, C_y, L_tot, x_A, x_B, x_C, max_M, min_M, x_max_M, x_min_M

# --- Eingabemaske in der Sidebar ---
st.sidebar.header("Geometrie (Längen in m)")
L_k1 = st.sidebar.number_input("Kragarm links", min_value=0.0, value=2.0, step=0.5)
L1   = st.sidebar.number_input("Feld 1", min_value=0.1, value=5.0, step=0.5)
# Feld 2 kann nun 0 sein für einen Einfeldträger
L2   = st.sidebar.number_input("Feld 2 (0 = Einfeldträger)", min_value=0.0, value=4.0, step=0.5)
L_k2 = st.sidebar.number_input("Kragarm rechts", min_value=0.0, value=1.5, step=0.5)

st.sidebar.header("Streckenlasten (in kN/m)")
q_k1 = st.sidebar.number_input("Last Kragarm links", value=5.0, step=1.0)
q1   = st.sidebar.number_input("Last Feld 1", value=10.0, step=1.0)
# Wenn Einfeldträger (L2=0), deaktiviere q2 visuell (wird ignoriert)
q2   = st.sidebar.number_input("Last Feld 2", value=8.0 if L2 > 0 else 0.0, step=1.0, disabled=(L2==0.0))
q_k2 = st.sidebar.number_input("Last Kragarm rechts", value=0.0, step=1.0)

st.sidebar.header("Punktlast (in kN)")
F   = st.sidebar.number_input("Größe der Punktlast F", value=25.0, step=5.0)
# Position ist jetzt relativ zu A. Negative Werte = links von A
st.sidebar.markdown("*Abstand relativ zu Auflager A (negativ = auf linkem Kragarm)*")
x_F_rel = st.sidebar.number_input("Position von F (von A)", min_value=-float(L_k1), max_value=float(L1+L2+L_k2), value=1.5, step=0.5)

# --- Berechnung ---
x, V, M, A_y, B_y, C_y, L_tot, x_A, x_B, x_C, max_M, min_M, x_max_M, x_min_M = berechne_traeger(L_k1, L1, L2, L_k2, q_k1, q1, q2, q_k2, F, x_F_rel)

# --- Ausgabe: Auflagerkräfte ---
st.subheader("Auflagerkräfte")
col1, col2, col3 = st.columns(3)
col1.metric("Auflager A", f"{A_y:.2f} kN")
col2.metric("Auflager B", f"{B_y:.2f} kN")
if L2 > 0:
    col3.metric("Auflager C", f"{C_y:.2f} kN")
else:
    col3.metric("Auflager C", "Entfällt (Einfeldträger)")

# --- Ausgabe: Extreme Momente ---
st.subheader("Maßgebende Biegemomente")
mcol1, mcol2 = st.columns(2)
mcol1.metric("Max. Feldmoment (Zug unten)", f"{max_M:.2f} kNm", f"bei {x_max_M:.2f} m von A", delta_color="off")
mcol2.metric("Max. Stützmoment (Zug oben)", f"{min_M:.2f} kNm", f"bei {x_min_M:.2f} m von A", delta_color="off")

# --- Grafische Darstellung ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2, 2]})

# Systembild
ax1.plot([0, L_tot], [0, 0], 'k-', linewidth=4)
ax1.plot(x_A, 0, '^', color='red', markersize=12, label='A')
ax1.plot(x_B, 0, '^', color='green', markersize=12, label='B')
if L2 > 0:
    ax1.plot(x_C, 0, '^', color='blue', markersize=12, label='C')

ax1.fill_between([0, x_A], 0, q_k1/25, color='gray', alpha=0.3)
ax1.fill_between([x_A, x_B], 0, q1/25, color='gray', alpha=0.3)
if L2 > 0:
    ax1.fill_between([x_B, x_C], 0, q2/25, color='gray', alpha=0.3)
    ax1.fill_between([x_C, L_tot], 0, q_k2/25, color='gray', alpha=0.3)
else:
    ax1.fill_between([x_B, L_tot], 0, q_k2/25, color='gray', alpha=0.3)

x_F_abs = x_A + x_F_rel
if F != 0:
    ax1.annotate('', xy=(x_F_abs, 0), xytext=(x_F_abs, 1.2), arrowprops=dict(facecolor='magenta', shrink=0.0, width=2, headwidth=8))
    ax1.text(x_F_abs, 1.3, f"F={F}kN", ha='center', color='magenta')

ax1.set_xlim(-0.5, L_tot + 0.5)
ax1.set_ylim(-0.5, 2.0)
ax1.axis('off')
title_str = "Zweifeldträger" if L2 > 0 else "Einfeldträger"
ax1.set_title(f"Statisches System und Belastung ({title_str})", fontweight='bold')

# Querkraft
ax2.plot(x, V, 'b-', linewidth=2)
ax2.fill_between(x, V, 0, color='blue', alpha=0.1)
ax2.axhline(0, color='k', linewidth=1)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylabel("Querkraft V [kN]", fontweight='bold')
ax2.set_xlim(0, L_tot)
ax2.invert_yaxis() 

# Biegemoment
ax3.plot(x, M, 'r-', linewidth=2)
ax3.fill_between(x, M, 0, color='red', alpha=0.1)
ax3.axhline(0, color='k', linewidth=1)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.set_ylabel("Biegemoment M [kNm]", fontweight='bold')
ax3.set_xlabel("Trägerlänge x [m]", fontweight='bold')
ax3.set_xlim(0, L_tot)
ax3.invert_yaxis() 

plt.tight_layout()
st.pyplot(fig)

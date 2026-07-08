import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import datetime
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

st.set_page_config(
    page_title="Électricité Intelligente",
    page_icon="⚡",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOGO
# ─────────────────────────────────────────────
logo_gif = Path(__file__).parent / "logo_animated.gif"
logo_png = Path(__file__).parent / "foto1.png"
if logo_gif.exists():
    st.image(str(logo_gif), width=230)
elif logo_png.exists():
    st.image(str(logo_png), width=230)

# ─────────────────────────────────────────────
# CARGA DE DATOS DESDE CSV
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = Path(__file__).parent / "fact_energie_extended.csv"
    df = pd.read_csv(csv_path)
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df["tranche_prix"] = pd.cut(
        df["prix_eur_mwh"],
        bins=[-float("inf"), 10, 30, 60, float("inf")],
        labels=["Très bon marché", "Bon marché", "Moyen", "Cher"]
    ).astype(str)
    return df

df = load_data()

# ─────────────────────────────────────────────
# RANDOM FOREST — entraînement
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    df_ml = _df.copy()
    df_ml["heure"]        = df_ml["date_heure"].dt.hour
    df_ml["mois"]         = df_ml["date_heure"].dt.month
    df_ml["jour_semaine"] = df_ml["date_heure"].dt.dayofweek
    df_ml["weekend"]      = (df_ml["jour_semaine"] >= 5).astype(int)

    features = ["heure", "mois", "jour_semaine", "weekend",
                "nucleaire", "eolien", "solaire", "hydraulique"]

    df_ml = df_ml.dropna(subset=features + ["prix_eur_mwh", "taux_co2"])
    X = df_ml[features]

    rf_prix = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_prix.fit(X, df_ml["prix_eur_mwh"])

    rf_co2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_co2.fit(X, df_ml["taux_co2"])

    avg_prod = df_ml.groupby(["heure", "mois"])[
        ["nucleaire", "eolien", "solaire", "hydraulique"]
    ].mean().reset_index()

    return rf_prix, rf_co2, avg_prod

rf_prix, rf_co2, avg_prod = train_models(df)

def predict_for_datetime(dt, df_source):
    heure        = dt.hour
    mois         = dt.month
    jour_semaine = dt.weekday()
    weekend      = 1 if jour_semaine >= 5 else 0

    target = pd.Timestamp(dt, tz="UTC")
    df_day = df_source[df_source["date_heure"].dt.date == dt.date()]

    if not df_day.empty:
        idx = (df_day["date_heure"] - target).abs().idxmin()
        row = df_day.loc[idx]
        nucleaire   = row["nucleaire"]
        eolien      = row["eolien"]
        solaire     = row["solaire"]
        hydraulique = row["hydraulique"]
    else:
        row_avg = avg_prod[
            (avg_prod["heure"] == heure) &
            (avg_prod["mois"]  == mois)
        ]
        if row_avg.empty:
            row_avg = avg_prod[avg_prod["heure"] == heure]
        if row_avg.empty:
            nucleaire, eolien, solaire, hydraulique = 40000, 5000, 3000, 8000
        else:
            row_avg     = row_avg.iloc[0]
            nucleaire   = row_avg["nucleaire"]
            eolien      = row_avg["eolien"]
            solaire     = row_avg["solaire"]
            hydraulique = row_avg["hydraulique"]

    X_pred = pd.DataFrame([{
        "heure": heure, "mois": mois,
        "jour_semaine": jour_semaine, "weekend": weekend,
        "nucleaire": nucleaire, "eolien": eolien,
        "solaire": solaire, "hydraulique": hydraulique
    }])

    prix_pred = rf_prix.predict(X_pred)[0]
    co2_pred  = rf_co2.predict(X_pred)[0]
    return round(prix_pred, 1), round(co2_pred, 0)

TRANCHE_DISPLAY = {
    "Cher":            ("À éviter",     "#ff2b2b"),
    "Très bon marché": ("Moment idéal", "#2bff5e"),
    "Bon marché":      ("Acceptable",   "#ffb02b"),
    "Moyen":           ("Acceptable",   "#ffb02b"),
}
MOIS_LABELS   = ["JAN","FÉV","MAR","AVR","MAI","JUN","JUL","AOÛ","SEP","OCT","NOV","DÉC"]
DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for k, v in [("fd",15),("fm",6),("fh",12),("fmin",0)]:
    if k not in st.session_state:
        st.session_state[k] = v

def change(field, delta):
    if field == "fd":
        max_d = DAYS_IN_MONTH[st.session_state.fm - 1]
        st.session_state.fd = (st.session_state.fd - 1 + delta) % max_d + 1
    elif field == "fm":
        st.session_state.fm = (st.session_state.fm - 1 + delta) % 12 + 1
        max_d = DAYS_IN_MONTH[st.session_state.fm - 1]
        if st.session_state.fd > max_d:
            st.session_state.fd = max_d
    elif field == "fh":
        st.session_state.fh  = (st.session_state.fh + delta) % 24
    elif field == "fmin":
        st.session_state.fmin = 30 if st.session_state.fmin == 0 else 0

def jump_to_tranche(tranche_db):
    safe_day = min(st.session_state.fd, DAYS_IN_MONTH[st.session_state.fm - 1])
    date_sel = datetime.date(2025, st.session_state.fm, safe_day)

    # Buscar en la fecha seleccionada
    df_day = df[df["date_heure"].dt.date == date_sel]
    df_t = df_day[df_day["tranche_prix"] == tranche_db]

    # Si no hay esa tranche en esa fecha → buscar en todo el mes
    if df_t.empty:
        df_mes = df[df["date_heure"].dt.month == st.session_state.fm]
        df_t = df_mes[df_mes["tranche_prix"] == tranche_db]

    # Último fallback → todo el dataset
    if df_t.empty:
        df_t = df[df["tranche_prix"] == tranche_db]

    if df_t.empty:
        return

    best = df_t.loc[df_t["prix_eur_mwh"].idxmin()]
    dt = best["date_heure"].to_pydatetime()

    # Actualizar DÍA, MES, HORA y MINUTO
    st.session_state.fd   = dt.day
    st.session_state.fm   = dt.month
    st.session_state.fh   = dt.hour
    st.session_state.fmin = 0 if dt.minute < 15 else 30

# ─────────────────────────────────────────────
# TIME CIRCUITS PANEL
# ─────────────────────────────────────────────
def render_time_circuits(df_source, present_dt, prix_pred, co2_pred):
    mois_fr = {
        "Jan":"JANV","Feb":"FÉVR","Mar":"MARS","Apr":"AVRIL",
        "May":"MAI","Jun":"JUIN","Jul":"JUIL","Aug":"AOÛT",
        "Sep":"SEPT","Oct":"OCT","Nov":"NOV","Dec":"DÉC"
    }
    target    = pd.Timestamp(present_dt, tz="UTC")
    df_day    = df_source[df_source["date_heure"].dt.date == present_dt.date()]
    if df_day.empty:
        df_day = df_source
    idx       = (df_day["date_heure"] - target).abs().idxmin()
    row_found = df_day.loc[idx]
    tranche_db_activa         = row_found["tranche_prix"]
    tranche_display_activa, _ = TRANCHE_DISPLAY.get(tranche_db_activa, ("Acceptable","#ffb02b"))

    def format_active(r):
        dt = r["date_heure"].to_pydatetime()
        return {
            "month": mois_fr[dt.strftime("%b")],
            "day":   dt.strftime("%d"),
            "year":  dt.strftime("%Y"),
            "hour":  dt.strftime("%H"),
            "min":   dt.strftime("%M"),
            "prix":  f"{r['prix_eur_mwh']:.1f}",
            "co2":   f"{r['taux_co2']:.0f}",
        }

    def format_zero():
        return {"month":"----","day":"00","year":"0000",
                "hour":"00","min":"00","prix":"00.0","co2":"000"}

    tranches_config = [
        ("À éviter",     "#ff2b2b"),
        ("Moment idéal", "#2bff5e"),
        ("Acceptable",   "#ffb02b"),
    ]

    def row_html(data, color, active):
        opacity = "1" if active else "0.2"
        label  = "color:#ffffff; font-family:'Courier New',monospace; font-size:18px; letter-spacing:4px; margin-bottom:8px; font-weight:bold;"
        number = f"font-family:'Digital-7 Mono','DSEG7 Classic','Courier New',monospace; font-size:62px; font-weight:bold; color:{color}; text-shadow:0 0 14px {color};"
        return f"""
        <div style="display:flex;align-items:center;justify-content:space-around;width:100%;opacity:{opacity};padding:10px 0;">
          <div style="text-align:center;"><div style="{label}">MOIS</div><div style="{number}">{data['month']}</div></div>
          <div style="text-align:center;"><div style="{label}">JOUR</div><div style="{number}">{data['day']}</div></div>
          <div style="text-align:center;"><div style="{label}">ANNÉE</div><div style="{number}">{data['year']}</div></div>
          <div style="text-align:center;color:#ffffff;font-family:'Courier New',monospace;font-size:16px;line-height:2;">▲<br>▼</div>
          <div style="text-align:center;"><div style="{label}">HEURE</div><div style="{number}">{data['hour']}</div></div>
          <div style="text-align:center;"><div style="{label}">MIN</div><div style="{number}">{data['min']}</div></div>
          <div style="width:2px;height:70px;background:#ffffff;opacity:0.2;margin:0 10px;align-self:center;"></div>
          <div style="text-align:center;"><div style="{label}">€ / MWh</div><div style="{number}">{data['prix']}</div></div>
          <div style="text-align:center;"><div style="{label}">g CO2</div><div style="{number}">{data['co2']}</div></div>
        </div>"""

    color_ia  = "#00ffff"
    dt_sel    = present_dt
    label_ia  = "color:#ffffff; font-family:'Courier New',monospace; font-size:18px; letter-spacing:4px; margin-bottom:8px; font-weight:bold;"
    number_ia = f"font-family:'Digital-7 Mono','DSEG7 Classic','Courier New',monospace; font-size:62px; font-weight:bold; color:{color_ia}; text-shadow:0 0 14px {color_ia};"

    ia_row = f"""
    <div style="display:flex;align-items:center;justify-content:space-around;width:100%;padding:10px 0;">
      <div style="text-align:center;"><div style="{label_ia}">MOIS</div><div style="{number_ia}">{mois_fr[dt_sel.strftime('%b')]}</div></div>
      <div style="text-align:center;"><div style="{label_ia}">JOUR</div><div style="{number_ia}">{dt_sel.strftime('%d')}</div></div>
      <div style="text-align:center;"><div style="{label_ia}">ANNÉE</div><div style="{number_ia}">{dt_sel.strftime('%Y')}</div></div>
      <div style="text-align:center;color:#ffffff;font-family:'Courier New',monospace;font-size:16px;line-height:2;">▲<br>▼</div>
      <div style="text-align:center;"><div style="{label_ia}">HEURE</div><div style="{number_ia}">{dt_sel.strftime('%H')}</div></div>
      <div style="text-align:center;"><div style="{label_ia}">MIN</div><div style="{number_ia}">{dt_sel.strftime('%M')}</div></div>
      <div style="width:2px;height:70px;background:#ffffff;opacity:0.2;margin:0 10px;align-self:center;"></div>
      <div style="text-align:center;"><div style="{label_ia}">€ / MWh</div><div style="{number_ia}">{prix_pred}</div></div>
      <div style="text-align:center;"><div style="{label_ia}">g CO2</div><div style="{number_ia}">{int(co2_pred)}</div></div>
    </div>"""

    rows_html = ""
    for tranche_name, color in tranches_config:
        is_active = (tranche_name == tranche_display_activa)
        data = format_active(row_found) if is_active else format_zero()
        rows_html += row_html(data, color, is_active)
        rows_html += f"""
        <div style="text-align:center;font-family:'Courier New',monospace;font-size:18px;
                    font-weight:bold;letter-spacing:4px;color:{color};
                    margin:4px 0 20px 0;opacity:{'1' if is_active else '0.2'};">
          {tranche_name.upper()}
        </div>
        <hr style="border:none;border-top:1px solid #222;margin:0 0 16px 0;">"""

    rows_html += ia_row
    rows_html += f"""
    <div style="text-align:center;font-family:'Courier New',monospace;font-size:18px;
                font-weight:bold;letter-spacing:4px;color:{color_ia};
                margin:4px 0 8px 0;">
       PRÉDICTION IA
    </div>"""

    html = f"""
    <style>@import url('https://fonts.cdnfonts.com/css/digital-7-mono');</style>
    <div style="background:#000000;border:3px solid #444;border-radius:10px;
                padding:30px 40px;width:100%;box-sizing:border-box;">
      {rows_html}
    </div>"""
    components.html(html, height=820, scrolling=False)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.cdnfonts.com/css/share-tech-mono');
.header-main {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 3.8vw !important;
    letter-spacing: 8px !important;
    font-weight: bold !important;
    color: #ffffff !important;
    text-align: right !important;
    text-shadow: 0 0 4px rgba(255,255,255,0.9), 0 0 10px rgba(255,255,255,0.4) !important;
    margin: 0 0 6px 0 !important;
    padding: 0 !important;
    line-height: 1.1 !important;
}
.header-sub {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 1.4vw !important;
    letter-spacing: 6px !important;
    color: rgba(255,255,255,0.55) !important;
    text-align: right !important;
    margin: 0 !important;
    padding: 0 !important;
}
.header-divider {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    margin: 24px 0 28px 0 !important;
}
.section-title {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 2.4vw !important;
    letter-spacing: 5px !important;
    font-weight: bold !important;
    color: #ffffff !important;
    text-shadow: 0 0 4px rgba(255,255,255,0.8), 0 0 12px rgba(255,255,255,0.3) !important;
    margin: 0 0 24px 0 !important;
    padding: 0 !important;
}
</style>
<p class="header-main">⚡ ÉLECTRICITÉ INTELLIGENTE</p>
<p class="header-sub">&gt; QUAND ET OÙ CONSOMMER EN FRANCE_</p>
<hr class="header-divider">
""", unsafe_allow_html=True)

st.markdown('<p class="section-title">🛠 PANNEAU DE CONTRÔLE TEMPOREL</p>', unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
div[data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    color: #444 !important;
    font-size: 32px !important;
    padding: 2px 0 !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: color 0.08s, transform 0.08s !important;
    line-height: 1 !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    color: #666 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button:active {
    color: #2bff5e !important;
    text-shadow: 0 0 14px #2bff5e, 0 0 28px #2bff5e !important;
    transform: scale(0.88) !important;
}
div[data-testid="stButton"] > button:focus {
    box-shadow: none !important;
    outline: none !important;
    color: #444 !important;
}
</style>
""", unsafe_allow_html=True)

def flip_card(value, label, font_size="56px"):
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;">
      <div style="color:#777;font-family:'Share Tech Mono','Courier New',monospace;
                  font-size:15px;letter-spacing:3px;margin-bottom:10px;
                  text-transform:uppercase;">{label}</div>
      <div style="background:#2a2a2a;border-radius:14px;width:110px;height:110px;
                  display:flex;align-items:center;justify-content:center;
                  font-family:'Share Tech Mono','Courier New',monospace;
                  font-size:{font_size};font-weight:bold;color:#ffb02b;
                  text-shadow:0 0 16px rgba(255,176,43,0.6),0 0 40px rgba(255,176,43,0.25);
                  box-shadow:0 6px 20px rgba(0,0,0,0.9),inset 0 1px 0 rgba(255,255,255,0.05);
                  position:relative;border:1px solid #383838;">
        <div style="position:absolute;left:6%;width:88%;height:1px;
                    background:rgba(0,0,0,0.55);top:50%;"></div>
        <span style="position:relative;z-index:1;">{value}</span>
      </div>
    </div>"""

c1, c2, c3, c4, c5, c6, cbtn = st.columns([1, 1, 0.1, 1, 0.2, 1, 1.2])

with c1:
    st.button("▲", key="up_day", on_click=change, args=("fd",  1))
    st.markdown(flip_card(f"{st.session_state.fd:02d}", "JOUR"), unsafe_allow_html=True)
    st.button("▼", key="dn_day", on_click=change, args=("fd", -1))

with c2:
    st.button("▲", key="up_mon", on_click=change, args=("fm",  1))
    st.markdown(flip_card(MOIS_LABELS[st.session_state.fm-1], "MOIS", font_size="34px"), unsafe_allow_html=True)
    st.button("▼", key="dn_mon", on_click=change, args=("fm", -1))

with c3:
    st.markdown("""<div style="height:150px;display:flex;align-items:center;justify-content:center;">
      <div style="width:1px;height:80px;background:linear-gradient(to bottom,transparent,#333,transparent);"></div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.button("▲", key="up_h",  on_click=change, args=("fh",   1))
    st.markdown(flip_card(f"{st.session_state.fh:02d}", "HEURE"), unsafe_allow_html=True)
    st.button("▼", key="dn_h",  on_click=change, args=("fh",  -1))

with c5:
    st.markdown("""<div style="height:150px;display:flex;align-items:center;justify-content:center;">
      <span style="font-size:44px;color:#555;font-family:'Share Tech Mono',monospace;
                   padding-top:20px;line-height:1;">:</span>
    </div>""", unsafe_allow_html=True)

with c6:
    st.button("▲", key="up_min", on_click=change, args=("fmin",  1))
    st.markdown(flip_card(f"{st.session_state.fmin:02d}", "MIN"), unsafe_allow_html=True)
    st.button("▼", key="dn_min", on_click=change, args=("fmin", -1))

with cbtn:
    if st.button("À ÉVITER",     key="btn_eviter"):
        jump_to_tranche("Cher")
        st.rerun()
    if st.button("MOMENT IDÉAL", key="btn_ideal"):
        jump_to_tranche("Très bon marché")
        st.rerun()
    if st.button("ACCEPTABLE",   key="btn_accept"):
        jump_to_tranche("Bon marché")
        st.rerun()

    components.html("""
    <script>
    (function() {
        const STYLES = {
            '\u00c0 \u00c9VITER': {color:'#ff2b2b',border:'1.5px solid rgba(255,43,43,0.6)',
                textShadow:'0 0 8px rgba(255,43,43,0.9)',boxShadow:'0 0 10px rgba(255,43,43,0.2)',
                background:'#1a0808',hoverBg:'#2e1212'},
            'MOMENT ID\u00c9AL': {color:'#2bff5e',border:'1.5px solid rgba(43,255,94,0.6)',
                textShadow:'0 0 8px rgba(43,255,94,0.9)',boxShadow:'0 0 10px rgba(43,255,94,0.2)',
                background:'#081a0e',hoverBg:'#0e2e18'},
            'ACCEPTABLE': {color:'#ffb02b',border:'1.5px solid rgba(255,176,43,0.6)',
                textShadow:'0 0 8px rgba(255,176,43,0.9)',boxShadow:'0 0 10px rgba(255,176,43,0.2)',
                background:'#1a1200',hoverBg:'#2e2010'}
        };
        function styleButtons(doc) {
            doc.querySelectorAll('[data-testid="stButton"] button').forEach(btn => {
                const p = btn.querySelector('p');
                const label = (p ? p.innerText : btn.innerText).trim().toUpperCase();
                const cfg = STYLES[label];
                if (!cfg) return;
                ['font-family','font-size','letter-spacing','font-weight','width','height',
                 'border-radius','padding','cursor','display','align-items','justify-content',
                 'transition','margin-bottom'].forEach(prop => {
                    const vals = {"font-family":"'Share Tech Mono',monospace","font-size":"13px",
                        "letter-spacing":"3px","font-weight":"bold","width":"100%","height":"52px",
                        "border-radius":"12px","padding":"0 8px","cursor":"pointer","display":"flex",
                        "align-items":"center","justify-content":"center","transition":"all 0.15s",
                        "margin-bottom":"8px"};
                    btn.style.setProperty(prop, vals[prop], 'important');
                });
                btn.style.setProperty('color', cfg.color, 'important');
                btn.style.setProperty('border', cfg.border, 'important');
                btn.style.setProperty('text-shadow', cfg.textShadow, 'important');
                btn.style.setProperty('box-shadow', cfg.boxShadow, 'important');
                btn.style.setProperty('background', cfg.background, 'important');
                if (p) {
                    p.style.setProperty('color', cfg.color, 'important');
                    p.style.setProperty('font-size', '13px', 'important');
                    p.style.setProperty('letter-spacing', '3px', 'important');
                    p.style.setProperty('font-weight', 'bold', 'important');
                    p.style.setProperty('text-shadow', cfg.textShadow, 'important');
                    p.style.setProperty('margin', '0', 'important');
                }
                if (!btn._styled) {
                    btn._styled = true;
                    btn.addEventListener('mouseenter', () => btn.style.setProperty('background', cfg.hoverBg, 'important'));
                    btn.addEventListener('mouseleave', () => btn.style.setProperty('background', cfg.background, 'important'));
                    btn.addEventListener('mousedown',  () => btn.style.setProperty('transform', 'scale(0.96)', 'important'));
                    btn.addEventListener('mouseup',    () => btn.style.setProperty('transform', 'scale(1)', 'important'));
                }
            });
        }
        function run() {
            try { styleButtons(window.parent.document); } catch(e) { styleButtons(document); }
        }
        run(); setTimeout(run,100); setTimeout(run,500); setTimeout(run,1500);
        try { new MutationObserver(run).observe(window.parent.document.body,{childList:true,subtree:true}); }
        catch(e) { new MutationObserver(run).observe(document.body,{childList:true,subtree:true}); }
    })();
    </script>
    """, height=0, scrolling=False)

# ─────────────────────────────────────────────
# RENDER PANEL
# ─────────────────────────────────────────────
safe_day = min(st.session_state.fd, DAYS_IN_MONTH[st.session_state.fm - 1])
date_voyage = datetime.date(2025, st.session_state.fm, safe_day)
present_dt_selected = datetime.datetime.combine(
    date_voyage, datetime.time(hour=st.session_state.fh, minute=st.session_state.fmin)
)

prix_pred, co2_pred = predict_for_datetime(present_dt_selected, df)
render_time_circuits(df, present_dt_selected, prix_pred, co2_pred)
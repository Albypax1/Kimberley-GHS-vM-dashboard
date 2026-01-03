# !pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from scipy import optimize
import requests

# ----------------- Constants & helpers -----------------
TWO_PI = 2*np.pi
LAT, LON = -28.7419, 24.7719  # Kimberley
TIMEZONE = "Africa/Johannesburg"

def wrap_angle(x):
    return np.mod(x, TWO_PI)

def exp_safe(expo):
    return np.exp(np.clip(expo, -700, 700))

def doy_today():
    today = pd.Timestamp.today(tz=TIMEZONE)
    return int(today.dayofyear)

def angle_to_days(angle):
    return (angle / TWO_PI) * 365.0

# ----------------- SS-GvM core -----------------

def normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu, n_grid=4096):
    grid = np.linspace(0, TWO_PI, n_grid, endpoint=False)
    expo = (
        k1*np.cos(grid-mu1)
        + k2*np.cos(2*(grid-mu2))
        + np.log(np.maximum(1 + eta*np.sin(grid-nu), 1e-12))
    )
    m = np.max(expo)
    return (TWO_PI/n_grid)*np.exp(m)*np.sum(np.exp(expo-m))

def logpdf_ssgvm(x, params):
    mu1, mu2, k1, k2, eta, nu = params
    Z = normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu)
    xw = wrap_angle(x)
    return (
        k1*np.cos(xw-mu1)
        + k2*np.cos(2*(xw-mu2))
        + np.log(np.maximum(1 + eta*np.sin(xw-nu), 1e-12))
        - np.log(Z)
    )

def fit_ssgvm_mle_all_starts(data, n_starts=30):
    data = wrap_angle(np.asarray(data))
    def neg_ll(p):
        val = np.sum(logpdf_ssgvm(data, p))
        return -val if np.isfinite(val) else 1e12
    bounds = optimize.Bounds([0,0,0,0,-0.999,0],[TWO_PI,TWO_PI,35.0,35.0,0.999,TWO_PI])
    best_x, best_fun = None, np.inf
    for _ in range(n_starts):
        init = np.array([
            np.random.rand()*TWO_PI, np.random.rand()*TWO_PI,
            np.random.gamma(2.0,1.0), np.random.gamma(2.0,1.0),
            np.tanh(np.random.randn()), np.random.rand()*TWO_PI
        ])
        res = optimize.minimize(neg_ll, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 6000})
        if res.fun < best_fun:
            best_fun, best_x = res.fun, res.x
    return best_x, -best_fun

# ----------------- Data ingestion (Open-Meteo) -----------------

def fetch_hotday_phases(start_date: str, end_date: str, q=0.90):
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": start_date, "end_date": end_date,
            "daily": ["temperature_2m_max"],
            "timezone": TIMEZONE
        }
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()["daily"]
        df = pd.DataFrame({"date": data["time"], "tmax": data["temperature_2m_max"]})
        df["date"] = pd.to_datetime(df["date"])
        df["doy"] = df["date"].dt.dayofyear
        thr = np.quantile(df["tmax"].dropna(), q)
        hot = df[df["tmax"] >= thr]
        phi = TWO_PI * (hot["doy"].values - 1) / 365.0
        return phi, df
    except Exception:
        # Fallback: monthly climatology-based synthetic phases
        monthly_tmax_F = np.array([93, 91, 88, 81, 75, 69, 69, 74, 82, 87, 90, 93], dtype=float)
        month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        mid_offsets = np.cumsum(np.concatenate([[0], month_lengths[:-1]])) + (month_lengths / 2.0)
        angles = (mid_offsets - 1) * TWO_PI / 365.0
        weights = monthly_tmax_F - monthly_tmax_F.min()
        weights = weights + 1e-6
        N_total = 2400
        rep_counts = np.maximum(1, (weights / weights.sum() * N_total).astype(int))
        phi = np.concatenate([np.repeat(angles[i], rep_counts[i]) for i in range(12)])
        df = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
        df["tmax"] = np.nan
        df["doy"] = df["date"].dt.dayofyear
        return phi, df

# ----------------- Streamlit UI -----------------

st.title("Kimberley SS‑GvM monitoring with treated water level forecast")

st.sidebar.header("Model configuration")
start_date = st.sidebar.date_input("Data start", date(2019,1,1))
end_date = st.sidebar.date_input("Data end", date.today())
hot_q = st.sidebar.slider("Hot-day quantile (threshold)", 0.80, 0.99, 0.90)
n_starts = st.sidebar.slider("MLE multi-starts", 10, 60, 30, step=5)

st.sidebar.header("Level forecast configuration")
current_level_m3 = st.sidebar.number_input("Current tank level (m³)", value=8000.0, min_value=0.0)
buffer_level_m3 = st.sidebar.number_input("Operational buffer level (m³)", value=6000.0, min_value=0.0)
tank_volume_m3 = st.sidebar.number_input("Tank nominal volume (m³)", value=10000.0, min_value=1.0)
production_m3pd = st.sidebar.number_input("Plant production (m³/day)", value=9500.0, min_value=0.0)
alpha_base = st.sidebar.number_input("Demand base α (m³/day)", value=8000.0, min_value=0.0)
beta_hot = st.sidebar.number_input("Demand sensitivity β (m³/day per HotProb)", value=4000.0, min_value=0.0)
gamma_evap = st.sidebar.number_input("Evap coefficient γ (m³/day per HotProb)", value=300.0, min_value=0.0)

st.sidebar.header("Alert thresholds")
high_density_thresh = st.sidebar.slider("High-density threshold (7-day sum)", 0.30, 1.20, 0.60, step=0.05)

# ----------------- Fit SS-GvM and build density -----------------

st.subheader("Data ingestion and SS‑GvM fit")
phi_samples, df_temp = fetch_hotday_phases(str(start_date), str(end_date), q=hot_q)
st.write(f"Hot-day samples: {len(phi_samples)}")
params, ll = fit_ssgvm_mle_all_starts(phi_samples, n_starts=n_starts)
mu1, mu2, k1, k2, eta, nu = params

days = np.arange(1, 366)
theta = 2*np.pi*days/365.0
Z = normalizer_ssgvm(*params)

density = np.exp(np.clip(k1*np.cos(theta-mu1) + k2*np.cos(2*(theta-mu2)), -700, 700)) \
          * np.maximum(1.0 + eta*np.sin(theta - nu), 1e-12) / Z

# Normalize to get per-day probabilities (sum over 365 = 1)
hotprob = density / density.sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(days, density, color="crimson", lw=2, label="SS‑GvM density")
ax.set_xlabel("Day of Year"); ax.set_ylabel("Probability density")
ax.set_title("SS‑GvM daily density (Kimberley)")
ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)

# ----------------- 7-day heat pressure index -----------------

st.subheader("7‑day heat pressure index")
doy = doy_today()
window = np.arange(doy, min(doy+7, 365+1))
heat_pressure = hotprob[window - 1].sum()
st.write(f"Today DOY: {doy}\n 7-day window: {window[0]}–{window[-1]}\n Heat pressure (sum HotProb): {heat_pressure:.3f}")

# ----------------- Treated water level forecast (next 7 days) -----------------

st.subheader("Treated water level forecast (next 7 days)")
forecast_days = 7
level_forecast = [current_level_m3]
demand_list, evap_list, hotprob_list = [], [], []
for d in range(forecast_days):
    idx = min(doy - 1 + d, 365 - 1)  # cap at 365
    hp = hotprob[idx]
    demand_t = alpha_base + beta_hot * hp
    evap_t = gamma_evap * hp
    next_level = level_forecast[-1] + (production_m3pd - demand_t - evap_t)
    level_forecast.append(next_level)
    demand_list.append(demand_t)
    evap_list.append(evap_t)
    hotprob_list.append(hp)
level_forecast = np.array(level_forecast[1:])  # drop initial

days_ahead = np.array([doy + i for i in range(forecast_days)])
days_ahead[days_ahead > 365] -= 365  # wrap into next year for labeling

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(np.arange(1, forecast_days+1), level_forecast, marker="o", lw=2, color="steelblue", label="Forecast level")
ax2.axhline(buffer_level_m3, color="red", ls="--", label="Buffer level")
ax2.set_xlabel("Days ahead"); ax2.set_ylabel("Level (m³)")
ax2.set_title("Forecasted treated water level (7 days)")
ax2.legend(); ax2.grid(alpha=0.3)
st.pyplot(fig2)

# Table of components

df_fore = pd.DataFrame({
    "DayAhead": np.arange(1, forecast_days+1),
    "CalendarDOY": days_ahead,
    "HotProb": np.round(hotprob_list, 5),
    "Demand_m3pd": np.round(demand_list, 1),
    "Evap_m3pd": np.round(evap_list, 1),
    "Level_m3": np.round(level_forecast, 1)
})

st.dataframe(df_fore)

# ----------------- Alerts and recommended actions -----------------

st.subheader("Alerts")
min_level = level_forecast.min()
within_15 = min_level <= buffer_level_m3 * (1 + 0.15)
within_10 = min_level <= buffer_level_m3 * (1 + 0.10)
breach_48h = (forecast_days >= 2) and (level_forecast[:2].min() <= buffer_level_m3)
current_breach = current_level_m3 <= buffer_level_m3
high_density = heat_pressure >= high_density_thresh

tier = "Normal"
reasons = []
if current_breach or breach_48h:
    tier = "Warning"
    reasons.append("Forecast breach ≤48h or current level ≤ buffer")
elif within_10 and high_density:
    tier = "Watch"
    reasons.append("Forecast within 10% of buffer with high heat pressure")
elif within_15:
    tier = "Advisory"
    reasons.append("Forecast within 15% of buffer")

st.write(f"Tier: {tier}")
if reasons:
    for r in reasons:
        st.error(r)
else:
    st.success("No immediate risks under current thresholds.")

st.subheader("Recommended actions")
if tier == "Warning":
    st.markdown("- **Production:** Increase output immediately; ensure CT compliance while ramping.\n- **Network:** Prioritize critical zones; schedule inter-reservoir transfers; throttle non-critical branches.\n- **Monitoring:** Hourly tank level and filter effluent; intensify leak hunt during night flow.\n- **Chemicals:** Prepare high-dose coagulation and PAC if water quality stress is concurrent.")
elif tier == "Watch":
    st.markdown("- **Production:** Pre-emptive output increase; stage staff and deliveries.\n- **Network:** Plan transfers to protect tank; adjust booster schedules to off-peak.\n- **Monitoring:** Tighten turbidity alarms; shorten filter runs.\n- **Comms:** Ready demand management messaging for peak hours.")
elif tier == "Advisory":
    st.markdown("- **Production:** Verify ramp capability; run jar tests to confirm dose curves.\n- **Network:** Check pump readiness; validate valve and transfer paths.\n- **Monitoring:** Calibrate instruments; track daily level vs forecast.")
else:
    st.markdown("- **Maintain normal operations.** Continue daily monitoring.")

# ----------------- Downloads -----------------

st.subheader("Downloads")
param_df = pd.DataFrame({"Param": ["mu1","mu2","k1","k2","eta","nu"], "Estimate": params})
csv_params = param_df.to_csv(index=False).encode("utf-8")
st.download_button("Download SS‑GvM parameters (CSV)", data=csv_params, file_name="kimberley_ssgvm_params.csv", mime="text/csv")

csv_forecast = df_fore.to_csv(index=False).encode("utf-8")
st.download_button("Download level forecast (CSV)", data=csv_forecast, file_name="treated_level_forecast_7d.csv", mime="text/csv")

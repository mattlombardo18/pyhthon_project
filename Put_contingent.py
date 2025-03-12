import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go

# =============================================================================
# Class Definitions
# =============================================================================

class Underlying:
    def __init__(self, spot, drift, sigma):
        self.spot = spot
        self.drift = drift
        self.sigma = sigma

class MC_simulation:
    def __init__(self, udl1, udl2, correl, NSimul, T):
        self.udl1 = udl1
        self.udl2 = udl2
        self.correl = correl
        self.NSimul = NSimul
        self.T = T

    def generate_gaussians(self):
        return np.random.multivariate_normal([0, 0],
                                             [[1, self.correl], [self.correl, 1]],
                                             self.NSimul).T

    def get_MC_distribution(self):
        g1, g2 = self.generate_gaussians()
        S1 = self.udl1.spot * np.exp((self.udl1.drift - 0.5 * self.udl1.sigma**2) * self.T +
                                     self.udl1.sigma * np.sqrt(self.T) * g1)
        S2 = self.udl2.spot * np.exp((self.udl2.drift - 0.5 * self.udl2.sigma**2) * self.T +
                                     self.udl2.sigma * np.sqrt(self.T) * g2)
        return S1, S2

class PutContingentOption:
    def __init__(self, Notional, Strike, Barrier, Currency, T):
        self.Notional = Notional
        self.Strike = Strike
        self.Barrier = Barrier
        self.Currency = Currency
        self.T = T

    def compute_payoff(self, s1, s2, discount_rate=None, T=None):
        if discount_rate is None:
            discount_rate = 0.025  # Default EUR rate
        if T is None:
            T = self.T
        # Payoff is active only if SX5E is below Strike.
        payoff = np.maximum(self.Strike - s1, 0)
        return np.exp(-discount_rate * T) * self.Notional * payoff

class Pricer:
    def __init__(self, MC, contract, discount_rate):
        self.MC = MC
        self.contract = contract
        self.discount_rate = discount_rate

    def compute_price(self):
        S1, S2 = self.MC.get_MC_distribution()
        # Option payoff is activated only when EUR/USD is above the barrier.
        payoffs = np.maximum(self.contract.Strike - S1, 0) * self.contract.Notional * (S2 > self.contract.Barrier)
        return np.mean(payoffs) * np.exp(-self.discount_rate * self.MC.T)

# =============================================================================
# Market Data Fetching (if needed)
# =============================================================================

def fetch_historical_data(tickers, start_date=None, end_date=None, period="1y", interval="1d"):
    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)["Close"]
    else:
        data = yf.download(tickers, period=period, interval=interval)["Close"]
    if data.empty:
        return None
    return data.ffill().dropna()

# =============================================================================
# Analytic Option Pricing and Finite Difference Delta Computation
# =============================================================================

def option_price_analytic(S_SX5E, S_FX, Strike, Barrier, r_EUR, r_USD, sigma_SX5E, sigma_FX, T):
    # The put on SX5E pays if SX5E is below Strike.
    if S_SX5E >= Strike:
        return 0.0
    d1 = (np.log(S_SX5E / Strike) + (r_EUR + 0.5 * sigma_SX5E**2) * T) / (sigma_SX5E * np.sqrt(T))
    d = (np.log(S_FX / Barrier) + (r_USD - r_EUR - 0.5 * sigma_FX**2) * T) / (sigma_FX * np.sqrt(T))
    prob_FX = 1 - norm.cdf(-d)  # risk-neutral probability that FX > Barrier at maturity
    price = (Strike - S_SX5E) * prob_FX * norm.cdf(-d1)
    return price

def compute_delta_SX5E_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E, sigma_FX, T, Strike, Barrier, shift=0.001):
    original_price = option_price_analytic(S_SX5E, S_FX, Strike, Barrier, r_EUR, r_USD, sigma_SX5E, sigma_FX, T)
    shifted_price = option_price_analytic(S_SX5E * (1 + shift), S_FX, Strike, Barrier, r_EUR, r_USD, sigma_SX5E, sigma_FX, T)
    delta = (shifted_price - original_price) / (shift * S_SX5E)
    return delta

def compute_delta_EURUSD_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E, sigma_FX, T, Strike, Barrier, shift=0.001):
    original_price = option_price_analytic(S_SX5E, S_FX, Strike, Barrier, r_EUR, r_USD, sigma_SX5E, sigma_FX, T)
    shifted_price = option_price_analytic(S_SX5E, S_FX * (1 + shift), Strike, Barrier, r_EUR, r_USD, sigma_SX5E, sigma_FX, T)
    delta = (shifted_price - original_price) / (shift * S_FX)
    return delta

def compute_delta_smoothing_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E, sigma_FX, T, Strike, Barrier):
    """
    Computes the delta of the contingent option and applies dynamic smoothing
    if FX is within ±1% of the barrier.
    """
    delta_SX5E = compute_delta_SX5E_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E, sigma_FX, T, Strike, Barrier)
    delta_FX = compute_delta_EURUSD_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E, sigma_FX, T, Strike, Barrier)
    smoothing_range = 0.01  # ±1% around the barrier
    lower_bound = Barrier * (1 - smoothing_range)
    upper_bound = Barrier * (1 + smoothing_range)
    if lower_bound <= S_FX <= upper_bound:
        delta_SX5E *= 0.5
        delta_FX *= 0.5
    return delta_SX5E, delta_FX

# =============================================================================
# Streamlit Interface Setup
# =============================================================================

st.set_page_config(page_title="Monte-Carlo Pricing", layout="wide")
st.markdown("<h1 style='text-align: center;'> Monte-Carlo Pricing of a Put Contingent Option</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

# =============================================================================
# Sidebar for Market Parameters
# =============================================================================

with st.sidebar:
    st.header("📊 Market Parameters")
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
    spot_SX5E = st.number_input("SX5E Spot", value=5000)
    spot_EURUSD = st.number_input("EUR/USD Spot", value=1.04)
    r_EUR = st.number_input("EUR Risk-Free Rate", value=0.025)
    r_USD = st.number_input("USD Risk-Free Rate", value=0.04)
    sigma_SX5E = st.number_input("SX5E Volatility", value=0.18)
    sigma_EURUSD = st.number_input("EUR/USD Volatility", value=0.08)
    T = st.number_input("Maturity (Years)", value=1.0)
    Notional = st.number_input("Notional (EUR)", value=100_000_000, format="%d")
    Strike = st.number_input("Strike Price", value=4600)
    Barrier = st.number_input("Barrier Level", value=1.10)
    correl = st.slider("Correlation between SX5E and EUR/USD", -1.0, 1.0, 0.3)
    NSimul = st.number_input("Number of Simulations", value=10_000, format="%d")

# =============================================================================
# Monte Carlo Simulation & Option Pricing
# =============================================================================

SX5E_obj = Underlying(spot_SX5E, 0.01, sigma_SX5E)
EURUSD_obj = Underlying(spot_EURUSD, r_USD - r_EUR, sigma_EURUSD)
MC = MC_simulation(SX5E_obj, EURUSD_obj, correl, NSimul, T)
contract = PutContingentOption(Notional, Strike, Barrier, "EUR", T)
pricer = Pricer(MC, contract, r_EUR)
price = pricer.compute_price()
S1, S2 = MC.get_MC_distribution()

# =============================================================================
# Simulation Verification and Asset Prices Summary
# =============================================================================
st.markdown("<h3 style='text-align: center;'>📊 Monte-Carlo Simulation Verification</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Asset Prices Summary")
    price_summary_df = pd.DataFrame({
        "Metric": ["Min Price", "Max Price"],
        "SX5E (Stock Index)": [f"{S1.min():,.2f}", f"{S1.max():,.2f}"],
        "EUR/USD (FX Rate)": [f"{S2.min():,.4f}", f"{S2.max():,.4f}"]
    })
    st.dataframe(price_summary_df.style.set_properties(**{"text-align": "center"}), use_container_width=True)
with col2:
    st.subheader("Payoff Activation Statistics")
    proportion_barrier = np.mean(S2 > Barrier) * 100
    proportion_payoff = np.mean((S2 > Barrier) & (S1 < Strike)) * 100
    payoff_summary_df = pd.DataFrame({
        "Metric": ["% EUR/USD Above Barrier", "% Payoff Activated"],
        "Value": [f"{proportion_barrier:.2f} %", f"{proportion_payoff:.2f} %"]
    })
    st.dataframe(payoff_summary_df.style.set_properties(**{"text-align": "center"}), use_container_width=True)

st.markdown("---", unsafe_allow_html=True)

# =============================================================================
# Descriptive Statistics of Simulations
# =============================================================================

df = pd.DataFrame({
    "S1 (SX5E)": S1,
    "S2 (EUR/USD)": S2,
    "Put Payoff": [contract.compute_payoff(s1, s2) for s1, s2 in zip(S1, S2)]
})
df["Put Payoff"] = pd.to_numeric(df["Put Payoff"], errors='coerce').fillna(0.0)
st.subheader("Simulation Descriptive Statistics")
st.dataframe(df.describe().applymap(lambda x: f"{x:,.2f}"), use_container_width=True)
st.markdown("---", unsafe_allow_html=True)

# =============================================================================
# Highlighting Positive Payoffs in Table
# =============================================================================

st.subheader("Highlighting Positive Payoffs")
styled_df = df.style.format({
    "S1 (SX5E)": "{:,.2f}",
    "S2 (EUR/USD)": "{:,.4f}",
    "Put Payoff": "{:,.2f}"
}).applymap(lambda x: 'background-color: #ffcccb' if x > 0 else '', subset=["Put Payoff"])
st.dataframe(styled_df, use_container_width=True)


# =============================================================================
# (A) Local Bump Delta Calculation (using the existing simulation paths)
# =============================================================================
def local_bump_delta_sx5e(S1, S2, shift, Strike, Barrier, Notional, discount_rate, T):
    """
    For each path i:
      - payoff_i = e^{-r*T} * Notional * max(Strike - S1_i, 0) * 1_{S2_i>Barrier}
      - payoff_up_i = e^{-r*T} * Notional * max(Strike - S1_i*(1+shift), 0) * 1_{S2_i>Barrier}
      - delta_i = (payoff_up_i - payoff_i) / [shift * S1_i]
    Then we average delta_i over all i.

    Returns:
      monetary_delta (Notional Delta)
      dimensionless_delta (Delta ignoring Notional)
    """
    discount_factor = np.exp(-discount_rate * T)

    payoff_base = discount_factor * Notional * np.maximum(Strike - S1, 0) * (S2 > Barrier)
    payoff_up = discount_factor * Notional * np.maximum(Strike - (S1 * (1 + shift)), 0) * (S2 > Barrier)

    diff = payoff_up - payoff_base
    delta_i = diff / (shift * S1)

    # 1) Full monetary delta
    monetary_delta = np.mean(delta_i)

    # 2) Multiply or not by Notional? 
    # Actually we've ALREADY multiplied by Notional in payoff_base & payoff_up.
    # So delta_i is in EUR per "1 unit" of S1. 
    # The average is still in EUR / S1. 
    # Typically for a dimensionless delta, we remove the Notional factor. 
    # So let's define dimensionless as if Notional=1.
    # We'll do that by re-running the same expression but ignoring Notional inside:
    payoff_base_dim = discount_factor * np.maximum(Strike - S1, 0) * (S2 > Barrier)
    payoff_up_dim = discount_factor * np.maximum(Strike - (S1 * (1 + shift)), 0) * (S2 > Barrier)
    diff_dim = payoff_up_dim - payoff_base_dim
    delta_i_dim = diff_dim / (shift * S1)
    dimensionless_delta = np.mean(delta_i_dim)

    # Because the 'diff' was already in real EUR terms, 
    # we keep the "monetary_delta" as is. 
    # The final step: multiply the monetary delta by 1 or by Notional?
    # Actually the "diff" is already in real EUR. 
    # So we do:
    monetary_delta = np.mean(diff) / (shift * np.mean(S1))  # This is one approach
    # Or simpler: just keep the average of delta_i 
    # which is in EUR / (S1_i). 
    # Typically we take S1 as the current spot for normalizing. 
    # We'll do that:
    monetary_delta = np.mean(diff) / (shift * spot_SX5E) 

    return monetary_delta, dimensionless_delta

def local_bump_delta_fx(S1, S2, shift, Strike, Barrier, Notional, discount_rate, T):
    """
    Similar local bump approach, but shifting final S2 by +shift% in each path.
    payoff_up_i = e^{-r*T} * Notional * max(Strike - S1_i, 0) * 1_{(S2_i*(1+shift))>Barrier}
    Then compute delta_i = (payoff_up_i - payoff_i) / [shift * S2_i].
    """
    discount_factor = np.exp(-discount_rate * T)

    payoff_base = discount_factor * Notional * np.maximum(Strike - S1, 0) * (S2 > Barrier)
    payoff_up = discount_factor * Notional * np.maximum(Strike - S1, 0) * ((S2 * (1 + shift)) > Barrier)

    diff = payoff_up - payoff_base
    delta_i = diff / (shift * S2)

    # 1) Full monetary delta
    # We'll keep the average but normalize by the current spot_EURUSD:
    monetary_delta = np.mean(diff) / (shift * spot_EURUSD)

    # 2) Dimensionless delta ignoring Notional
    payoff_base_dim = discount_factor * np.maximum(Strike - S1, 0) * (S2 > Barrier)
    payoff_up_dim = discount_factor * np.maximum(Strike - S1, 0) * ((S2 * (1 + shift)) > Barrier)
    diff_dim = payoff_up_dim - payoff_base_dim
    delta_i_dim = diff_dim / (shift * S2)
    dimensionless_delta = np.mean(delta_i_dim)

    return monetary_delta, dimensionless_delta

# ------------------------------------------------------------------------------
# (B) Actually compute these Deltas
# ------------------------------------------------------------------------------
shift_size = 0.01  # 1% local bump
delta_sx5e_notional, delta_sx5e_dim = local_bump_delta_sx5e(
    S1, S2, shift_size, Strike, Barrier, Notional, r_EUR, T
)
delta_fx_notional, delta_fx_dim = local_bump_delta_fx(
    S1, S2, shift_size, Strike, Barrier, Notional, r_EUR, T
)

# ------------------------------------------------------------------------------
# (C) Harmonized Display of Price & Both Deltas
# ------------------------------------------------------------------------------
st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'> Put Contingent Option Price & MC Deltas</h2>", unsafe_allow_html=True)
st.markdown(
    f"<h3 style='text-align: center;'>Option Price (EUR): "
    f"<strong style='color: #008000;'>{price:,.2f}</strong></h3>",
    unsafe_allow_html=True
)

st.markdown("---", unsafe_allow_html=True)

delta_df = pd.DataFrame({
    "Greek": [
        "Delta SX5E (Dimensionless)",
        "Delta SX5E (Notional)",
        "Delta EUR/USD (Dimensionless)",
        "Delta EUR/USD (Notional)"
    ],
    "Value": [
        f"{delta_sx5e_dim:.4f}",
        f"{delta_sx5e_notional:,.2f}",
        f"{delta_fx_dim:.4f}",
        f"{delta_fx_notional:,.2f}"
    ]
})
st.dataframe(delta_df.style.set_properties(**{"text-align": "center"}), use_container_width=True)

st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)


# =============================================================================
# Load Historical Data and Compute Rolling Metrics
# =============================================================================

tickers = ["^STOXX50E", "EURUSD=X"]
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=365)

data = fetch_historical_data(tickers, start_date=start_date, end_date=end_date)

if data is None:
    st.warning("⚠️ Data could not be fetched. Check ticker symbols or internet connection.")
else:
    # Rename columns for clarity
    data = data.rename(columns={"^STOXX50E": "SX5E", "EURUSD=X": "EURUSD"})

    # Ensure business days and forward-fill missing values
    business_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
    data = data.reindex(business_days).ffill()

    # Extract historical price arrays
    historical_sx5e = data["SX5E"].to_numpy()
    historical_eurusd = data["EURUSD"].to_numpy()
    dates = pd.to_datetime(data.index)

    # Compute Log Returns
    log_returns = np.log(data / data.shift(1)).dropna()

    # Compute Rolling Volatility (30-day window)
    rolling_sigma_sx5e = log_returns["SX5E"].rolling(30).std() * np.sqrt(252)
    rolling_sigma_fx = log_returns["EURUSD"].rolling(30).std() * np.sqrt(252)

    # Ensure no NaN values in rolling metrics
    rolling_sigma_sx5e, rolling_sigma_fx = rolling_sigma_sx5e.dropna(), rolling_sigma_fx.dropna()
    
    valid_dates = rolling_sigma_sx5e.index.intersection(rolling_sigma_fx.index)

    # Compute Rolling Correlation (30-day window)
    rolling_correlation = log_returns["SX5E"].rolling(30).corr(log_returns["EURUSD"])
    historical_correlation = log_returns["SX5E"].corr(log_returns["EURUSD"])

    # =============================================================================
    # Plot Historical Prices for SX5E & EUR/USD with Active Period Highlighting
    # =============================================================================
    
    fig = go.Figure()
    
    # Plot Historical SX5E Prices
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["SX5E"],
        mode="lines",
        name="Historical SX5E",
        line=dict(color='#1F77B4', width=2),
        yaxis="y1"
    ))
    
    # Plot Historical EUR/USD Prices
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["EURUSD"],
        mode="lines",
        name="Historical EUR/USD",
        line=dict(color='#FF7F0E', width=2),
        yaxis="y2"
    ))
    
    # -----------------------
    # Dynamic Axis Ranges
    # -----------------------
    sx5e_min = min(data["SX5E"].min(), Strike) * 0.95
    sx5e_max = max(data["SX5E"].max(), Strike) * 1.05
    eurusd_min = min(data["EURUSD"].min(), Barrier) * 0.98
    eurusd_max = max(data["EURUSD"].max(), Barrier) * 1.02
    
    fig.add_trace(go.Scatter(
        x=[data.index.min(), data.index.max()],
        y=[Strike, Strike],
        mode="lines",
        name=f"SX5E Strike: {Strike}",
        line=dict(color="red", dash="dash", width=2),
        hoverinfo="skip",
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=[data.index.min(), data.index.max()],
        y=[Barrier, Barrier],
        mode="lines",
        name=f"EUR/USD Barrier: {Barrier}",
        line=dict(color="green", dash="dash", width=2),
        hoverinfo="skip",
        yaxis="y2"
    ))
    
    afficher_active_period = True
    
    if afficher_active_period:
        active_condition = (data["EURUSD"] > Barrier) & (data["SX5E"] < Strike)
        active_dates = data.index[active_condition]
    
        active_groups = []
        if not active_dates.empty:
            group_start = active_dates[0]
            group_end = active_dates[0]
            for current_date in active_dates[1:]:
                if (current_date - group_end).days <= 1:
                    group_end = current_date
                else:
                    active_groups.append((group_start, group_end))
                    group_start = current_date
                    group_end = current_date
            active_groups.append((group_start, group_end))
    
        for start, end in active_groups:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                y0=0,
                x1=end,
                y1=1,
                fillcolor="purple",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
    
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="purple", opacity=0.2),
            name="Active Period"
        ))
    
    # -----------------------
    # Configure Layout & Styling
    # -----------------------
    fig.update_layout(
        title="Historical Prices for SX5E and EUR/USD with Active Period Highlighting",
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="LightGrey"
        ),
        yaxis=dict(
            title="SX5E Price",
            side="left",
            range=[sx5e_min, sx5e_max],
            showgrid=True,
            gridcolor="LightGrey"
        ),
        yaxis2=dict(
            title="EUR/USD Price",
            overlaying="y",
            anchor="x",
            side="right",
            range=[eurusd_min, eurusd_max],
            showgrid=False
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=60, b=50, t=50)
    )
    
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Option Price vs. Correlation Sensitivity Analysis
    # =============================================================================

    correl_values = np.linspace(-0.5, 0.5, 20)
    def price_for_correlation(c):
        return Pricer(MC_simulation(SX5E_obj, EURUSD_obj, c, NSimul, T), contract, r_EUR).compute_price()
    prices = np.vectorize(price_for_correlation)(correl_values)
    correlation_df = pd.DataFrame({
        "Correlation": correl_values,
        "Option Price": prices
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=correlation_df["Correlation"],
        y=correlation_df["Option Price"],
        mode="lines+markers",
        name="Option Price",
        marker=dict(size=6, color='#2C3E50'),
        line=dict(color='#2C3E50', width=2)
    ))
    fig.update_layout(
        title="Put Contingent Option Price vs. Correlation",
        xaxis=dict(title=dict(text="Correlation between SX5E and EUR/USD"), showgrid=True),
        yaxis=dict(title=dict(text="Option Price (EUR)"), showgrid=True),
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Rolling 30-Day Correlation Plot
    # =============================================================================

    rolling_corr_df = pd.DataFrame({
        "Date": rolling_correlation.index,
        "Rolling Correlation": rolling_correlation.values
    }).set_index("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_corr_df.index,
        y=rolling_corr_df["Rolling Correlation"],
        mode="lines",
        name="Rolling 30-day Correlation",
        line=dict(color="red"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=rolling_corr_df.index,
        y=[historical_correlation] * len(rolling_corr_df),
        mode="lines",
        name=f"Historical Correlation: {historical_correlation:.2f}",
        line=dict(color="black", dash="dash"),
        yaxis="y1"
    ))
    fig.update_layout(
        title="Rolling 30-Day Correlation between SX5E and EUR/USD",
        xaxis=dict(title=dict(text="Date")),
        yaxis=dict(
            title=dict(text="Correlation"),
            range=[-1, 1],
            side="left",
            showgrid=True
        ),
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Compute Option Price and Delta Evolution Over Historical Dates
    # =============================================================================

    historical_prices = []
    historical_deltas_SX5E = []
    historical_deltas_FX = []
    final_dates = []
    for date in valid_dates:
        try:
            S_SX5E = data.loc[date, "SX5E"]
            S_FX = data.loc[date, "EURUSD"]
            sigma_SX5E_val = rolling_sigma_sx5e.loc[date]
            sigma_FX_val = rolling_sigma_fx.loc[date]
            # Compute analytic option price using the new function
            put_price = option_price_analytic(S_SX5E, S_FX, Strike, Barrier, r_EUR, r_USD, sigma_SX5E_val, sigma_FX_val, T)
            # Compute finite difference deltas for SX5E and EUR/USD
            delta_SX5E = compute_delta_SX5E_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E_val, sigma_FX_val, T, Strike, Barrier)
            delta_FX = compute_delta_EURUSD_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E_val, sigma_FX_val, T, Strike, Barrier)
            historical_prices.append(put_price)
            historical_deltas_SX5E.append(delta_SX5E)
            historical_deltas_FX.append(delta_FX)
            final_dates.append(date)
        except KeyError:
            continue

    results_df = pd.DataFrame({
        "Date": final_dates,
        "SX5E_Price": data.loc[final_dates, "SX5E"].values,
        "EURUSD_Price": data.loc[final_dates, "EURUSD"].values,
        "Put_Price": historical_prices,
        "Delta_SX5E": historical_deltas_SX5E,
        "Delta_FX": historical_deltas_FX
    }).set_index("Date")

    # =============================================================================
    # Plot Delta Evolution & Option Price
    # =============================================================================

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df["Delta_SX5E"],
        mode="lines",
        name="Delta SX5E",
        line=dict(color="blue"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df["Delta_FX"],
        mode="lines",
        name="Delta FX",
        line=dict(color="red"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df["Put_Price"],
        mode="lines",
        name="Put Option Price",
        line=dict(color="green", dash="dash"),
        yaxis="y2"
    ))
    fig.update_layout(
        title="Delta Evolution for SX5E and EUR/USD + Put Option Price",
        xaxis=dict(title=dict(text="Date")),
        yaxis=dict(
            title=dict(text="Finite Difference Delta"),
            range=[-1, 1],
            side="left",
            showgrid=False
        ),
        yaxis2=dict(
            title=dict(text="Put Option Price (EUR)"),
            overlaying="y",
            anchor="x",
            side="right",
            showgrid=False
        ),
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Compute One-Year P&L of the Delta-Hedged Structure using Finite Difference Deltas
    # =============================================================================

    daily_pnl = []
    hedge_pnl = []
    option_pnl = []
    dates_pnl = []
    previous_delta_sx5e = historical_deltas_SX5E[0]
    previous_delta_fx = historical_deltas_FX[0]
    for i in range(1, len(final_dates)):
        date = final_dates[i]
        prev_sx5e = results_df.loc[final_dates[i-1], "SX5E_Price"]
        curr_sx5e = results_df.loc[date, "SX5E_Price"]
        prev_fx = results_df.loc[final_dates[i-1], "EURUSD_Price"]
        curr_fx = results_df.loc[date, "EURUSD_Price"]
        prev_option_price = results_df.loc[final_dates[i-1], "Put_Price"]
        curr_option_price = results_df.loc[date, "Put_Price"]
        current_delta_sx5e = results_df.loc[date, "Delta_SX5E"]
        current_delta_fx = results_df.loc[date, "Delta_FX"]
        sx5e_hedge_pnl = (curr_sx5e - prev_sx5e) * previous_delta_sx5e * Notional
        fx_hedge_pnl = (curr_fx - prev_fx) * previous_delta_fx * Notional
        option_profit = (curr_option_price - prev_option_price) * Notional
        total_hedge_pnl = sx5e_hedge_pnl + fx_hedge_pnl
        daily_pnl.append(option_profit - total_hedge_pnl)
        hedge_pnl.append(total_hedge_pnl)
        option_pnl.append(option_profit)
        dates_pnl.append(date)
        previous_delta_sx5e = current_delta_sx5e
        previous_delta_fx = current_delta_fx

    pnl_df = pd.DataFrame({
        "Date": dates_pnl,
        "Total P&L": daily_pnl,
        "Hedge P&L": hedge_pnl,
        "Option P&L": option_pnl
    }).set_index("Date")

    # =============================================================================
    # Plot P&L Over Time
    # =============================================================================

    st.subheader("📈 One-Year P&L of the Delta-Hedged Structure")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl_df.index, y=pnl_df["Total P&L"],
        mode="lines", name="Total P&L", line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df.index, y=pnl_df["Hedge P&L"],
        mode="lines", name="Hedge P&L", line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df.index, y=pnl_df["Option P&L"],
        mode="lines", name="Option P&L", line=dict(color="blue", dash="dash")
    ))
    fig.update_layout(
        title="P&L Evolution of the Delta-Hedged Structure",
        xaxis=dict(title=dict(text="Date")),
        yaxis=dict(title=dict(text="Profit & Loss (EUR)")),
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark"
    )
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
    st.markdown("### **__Cumulative P&L Summary__**")
    st.dataframe(pnl_df.describe(), width=800)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Repricing Using Vanilla Spreads (ε = 1%)
    # =============================================================================

    def black_scholes_call(S, K, r, sigma, T):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    epsilon = 0.01
    K1 = Barrier  
    K2 = Barrier * (1 + epsilon)
    def compute_digital_approximation(S_FX, r_EUR, sigma_FX, T):
        call_1 = black_scholes_call(S_FX, K1, r_EUR, sigma_FX, T)
        call_2 = black_scholes_call(S_FX, K2, r_EUR, sigma_FX, T)
        return call_1 - call_2
    def compute_price_with_smoothing():
        S1, S2 = MC.get_MC_distribution()
        digital_approx = np.array([compute_digital_approximation(s2, r_EUR, sigma_EURUSD, T) for s2 in S2])
        payoff_vector = np.maximum(Strike - S1, 0) * digital_approx
        return np.mean(payoff_vector) * Notional * np.exp(-r_EUR * T)
    smoothed_price = compute_price_with_smoothing()

    # =============================================================================
    # Compute New Deltas with Smoothing using Finite Difference & Vanilla Spread Approximation
    # =============================================================================

    new_deltas_SX5E, new_deltas_FX = [], []
    for date in final_dates:
        try:
            S_SX5E = data.loc[date, "SX5E"]
            S_FX = data.loc[date, "EURUSD"]
            sigma_SX5E_val = rolling_sigma_sx5e.loc[date]
            sigma_FX_val = rolling_sigma_fx.loc[date]
            delta_SX5E, delta_FX = compute_delta_smoothing_finite(S_SX5E, S_FX, r_EUR, r_USD, sigma_SX5E_val, sigma_FX_val, T, Strike, Barrier)
            new_deltas_SX5E.append(delta_SX5E)
            new_deltas_FX.append(delta_FX)
        except KeyError:
            continue

    results_df["New_Delta_SX5E"] = new_deltas_SX5E
    results_df["New_Delta_FX"] = new_deltas_FX
    st.subheader("New Deltas After Smoothing with Vanilla Spread Approximation")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df["New_Delta_SX5E"],
        mode="lines",
        name="New Delta SX5E",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df["New_Delta_FX"],
        mode="lines",
        name="New Delta FX",
        line=dict(color="red", dash="dash")
    ))
    fig.update_layout(
        title="New Delta Evolution After Smoothing",
        xaxis=dict(title=dict(text="Date")),
        yaxis=dict(title=dict(text="Delta")),
        template="plotly_dark"
    )
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> Repriced Put Contingent Option Using Vanilla Spread Approximation</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style='text-align: center;'>New Option Price (EUR): <strong style="color: #008000;">{smoothed_price:,.2f}</strong></h3>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Compare Cumulated P&L of Initial and Smoothed Structures
    # =============================================================================

    daily_pnl_smooth = []
    hedge_pnl_smooth = []
    option_pnl_smooth = []
    dates_pnl_smooth = []
    previous_smooth_delta_sx5e = new_deltas_SX5E[0]
    previous_smooth_delta_fx = new_deltas_FX[0]
    for i in range(1, len(final_dates)):
        date = final_dates[i]
        prev_sx5e = results_df.loc[final_dates[i-1], "SX5E_Price"]
        curr_sx5e = results_df.loc[date, "SX5E_Price"]
        prev_fx = results_df.loc[final_dates[i-1], "EURUSD_Price"]
        curr_fx = results_df.loc[date, "EURUSD_Price"]
        prev_option_price = results_df.loc[final_dates[i-1], "Put_Price"]
        curr_option_price = results_df.loc[date, "Put_Price"]
        current_smooth_delta_sx5e = results_df.loc[date, "New_Delta_SX5E"]
        current_smooth_delta_fx = results_df.loc[date, "New_Delta_FX"]
        sx5e_hedge_pnl_smooth = (curr_sx5e - prev_sx5e) * previous_smooth_delta_sx5e * Notional
        fx_hedge_pnl_smooth = (curr_fx - prev_fx) * previous_smooth_delta_fx * Notional
        option_profit_smooth = (curr_option_price - prev_option_price) * Notional
        total_hedge_pnl_smooth = sx5e_hedge_pnl_smooth + fx_hedge_pnl_smooth
        daily_pnl_smooth.append(option_profit_smooth - total_hedge_pnl_smooth)
        hedge_pnl_smooth.append(total_hedge_pnl_smooth)
        option_pnl_smooth.append(option_profit_smooth)
        dates_pnl_smooth.append(date)
        previous_smooth_delta_sx5e = current_smooth_delta_sx5e
        previous_smooth_delta_fx = current_smooth_delta_fx

    pnl_df_smooth = pd.DataFrame({
        "Date": dates_pnl_smooth,
        "Total P&L Smoothed": np.cumsum(daily_pnl_smooth),
        "Hedge P&L Smoothed": np.cumsum(hedge_pnl_smooth),
        "Option P&L Smoothed": np.cumsum(option_pnl_smooth)
    }).set_index("Date")

    pnl_df_initial = pd.DataFrame({
        "Date": dates_pnl,
        "Total P&L Initial": np.cumsum(daily_pnl),
        "Hedge P&L Initial": np.cumsum(hedge_pnl),
        "Option P&L Initial": np.cumsum(option_pnl)
    }).set_index("Date")

    st.subheader("Cumulative P&L Comparison: Initial vs. Smoothed Structure")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl_df_initial.index, y=pnl_df_initial["Total P&L Initial"],
        mode="lines", name="Total P&L (Initial)", line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df_smooth.index, y=pnl_df_smooth["Total P&L Smoothed"],
        mode="lines", name="Total P&L (Smoothed)", line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df_initial.index, y=pnl_df_initial["Hedge P&L Initial"],
        mode="lines", name="Hedge P&L (Initial)", line=dict(color="orange", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df_smooth.index, y=pnl_df_smooth["Hedge P&L Smoothed"],
        mode="lines", name="Hedge P&L (Smoothed)", line=dict(color="blue", dash="dash")
    ))
    fig.update_layout(
        title="Cumulative P&L: Initial vs. Smoothed Hedging",
        xaxis=dict(title=dict(text="Date")),
        yaxis=dict(title=dict(text="Cumulative P&L (EUR)")),
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark"
    )
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
    st.subheader("Summary of Cumulative P&L")
    pnl_summary = pd.DataFrame({
        "Metric": ["Final P&L (Initial)", "Final P&L (Smoothed)",
                   "Final Hedge P&L (Initial)", "Final Hedge P&L (Smoothed)"],
        "Value (EUR)": [pnl_df_initial["Total P&L Initial"].iloc[-1],
                        pnl_df_smooth["Total P&L Smoothed"].iloc[-1],
                        pnl_df_initial["Hedge P&L Initial"].iloc[-1],
                        pnl_df_smooth["Hedge P&L Smoothed"].iloc[-1]]
    })
    st.dataframe(pnl_summary.style.format({"Value (EUR)": "{:,.2f}"}), width=500)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)

    # =============================================================================
    # Compare Three Hedging Strategies
    # =============================================================================

    scaling_factor = 2  
    pnl_strategy_1 = []  # No smoothing (Standard Digital)
    pnl_strategy_2 = []  # Smoothed delta hedge
    pnl_strategy_3 = []  # Scaled delta hedge
    dates_strategy = []
    prev_delta_1_sx5e, prev_delta_1_fx = historical_deltas_SX5E[0], historical_deltas_FX[0]
    prev_delta_2_sx5e, prev_delta_2_fx = new_deltas_SX5E[0], new_deltas_FX[0]
    prev_delta_3_sx5e, prev_delta_3_fx = prev_delta_1_sx5e * scaling_factor, prev_delta_1_fx * scaling_factor
    for i in range(1, len(final_dates)):
        date = final_dates[i]
        prev_sx5e = results_df.loc[final_dates[i-1], "SX5E_Price"]
        curr_sx5e = results_df.loc[date, "SX5E_Price"]
        prev_fx = results_df.loc[final_dates[i-1], "EURUSD_Price"]
        curr_fx = results_df.loc[date, "EURUSD_Price"]
        prev_price = results_df.loc[final_dates[i-1], "Put_Price"]
        curr_price = results_df.loc[date, "Put_Price"]
        delta_1_sx5e, delta_1_fx = results_df.loc[date, "Delta_SX5E"], results_df.loc[date, "Delta_FX"]
        delta_2_sx5e, delta_2_fx = results_df.loc[date, "New_Delta_SX5E"], results_df.loc[date, "New_Delta_FX"]
        delta_3_sx5e, delta_3_fx = delta_1_sx5e * scaling_factor, delta_1_fx * scaling_factor
        hedge_pnl_1 = (curr_sx5e - prev_sx5e) * prev_delta_1_sx5e * Notional + (curr_fx - prev_fx) * prev_delta_1_fx * Notional
        hedge_pnl_2 = (curr_sx5e - prev_sx5e) * prev_delta_2_sx5e * Notional + (curr_fx - prev_fx) * prev_delta_2_fx * Notional
        hedge_pnl_3 = (curr_sx5e - prev_sx5e) * prev_delta_3_sx5e * Notional + (curr_fx - prev_fx) * prev_delta_3_fx * Notional
        option_pnl_val = (curr_price - prev_price) * Notional
        pnl_strategy_1.append(option_pnl_val - hedge_pnl_1)
        pnl_strategy_2.append(option_pnl_val - hedge_pnl_2)
        pnl_strategy_3.append(option_pnl_val - hedge_pnl_3)
        dates_strategy.append(date)
        prev_delta_1_sx5e, prev_delta_1_fx = delta_1_sx5e, delta_1_fx
        prev_delta_2_sx5e, prev_delta_2_fx = delta_2_sx5e, delta_2_fx
        prev_delta_3_sx5e, prev_delta_3_fx = delta_3_sx5e, delta_3_fx

    pnl_df_strategies = pd.DataFrame({
        "Date": dates_strategy,
        "Strategy 1 P&L": np.cumsum(pnl_strategy_1),
        "Strategy 2 P&L": np.cumsum(pnl_strategy_2),
        "Strategy 3 P&L": np.cumsum(pnl_strategy_3)
    }).set_index("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl_df_strategies.index, y=pnl_df_strategies["Strategy 1 P&L"],
        mode="lines", name="Strategy 1: No Smoothing", line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df_strategies.index, y=pnl_df_strategies["Strategy 2 P&L"],
        mode="lines", name="Strategy 2: Hedging with Smoothing", line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df_strategies.index, y=pnl_df_strategies["Strategy 3 P&L"],
        mode="lines", name="Strategy 3: Scaling Factor Hedge", line=dict(color="blue")
    ))
    fig.update_layout(
        title="Strategy P&L Comparison",
        xaxis=dict(title=dict(text="Date")),
        yaxis=dict(title=dict(text="Cumulative P&L (EUR)")),
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark"
    )
    st.plotly_chart(fig)
    st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
    st.subheader("Final P&L Comparison of the Strategies")
    pnl_summary_strategies = pd.DataFrame({
        "Strategy": ["Strategy 1: No Smoothing", "Strategy 2: Hedging with Smoothing", "Strategy 3: Scaling Factor"],
        "Final P&L (EUR)": [
            pnl_df_strategies["Strategy 1 P&L"].iloc[-1],
            pnl_df_strategies["Strategy 2 P&L"].iloc[-1],
            pnl_df_strategies["Strategy 3 P&L"].iloc[-1]
        ]
    })
    st.dataframe(pnl_summary_strategies.style.format({"Final P&L (EUR)": "{:,.2f}"}), width=500)

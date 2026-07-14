# ============================================================
# AdIntel
# Intelligent Sales Prediction & Advertising Analytics
# Single-file Streamlit Application
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AdIntel | Advertising Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_advertising.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.pkl"

REPORTS_DIR = PROJECT_ROOT / "reports"

STREAMLIT_CONFIG_PATH = REPORTS_DIR / "streamlit_config.json"
BUSINESS_SUMMARY_PATH = REPORTS_DIR / "business_summary.json"
SCENARIO_PATH = REPORTS_DIR / "scenario_comparison.csv"
SENSITIVITY_PATH = REPORTS_DIR / "channel_sensitivity_analysis.csv"
REALLOCATION_PATH = REPORTS_DIR / "budget_reallocation_analysis.csv"
CHANNEL_INSIGHTS_PATH = REPORTS_DIR / "channel_insights.csv"
RECOMMENDATIONS_PATH = REPORTS_DIR / "business_recommendations.csv"
MODEL_METRICS_PATH = REPORTS_DIR / "tuned_model_metrics.csv"
CV_RESULTS_PATH = REPORTS_DIR / "tuned_cross_validation_results.csv"
FINAL_PREDICTIONS_PATH = REPORTS_DIR / "final_model_predictions.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
BASELINE_TUNED_PATH = REPORTS_DIR / "baseline_vs_tuned_models.csv"


# ============================================================
# Professional Light Theme
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --background: #F6F8FC;
        --surface: #FFFFFF;
        --primary: #2563EB;
        --primary-dark: #1D4ED8;
        --primary-soft: #EFF6FF;
        --navy: #0F172A;
        --text: #1E293B;
        --muted: #64748B;
        --border: #E2E8F0;
        --success: #15803D;
        --success-soft: #F0FDF4;
        --danger: #B91C1C;
        --danger-soft: #FEF2F2;
        --warning: #B45309;
        --warning-soft: #FFFBEB;
    }

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    .sidebar-brand {
        padding: 0.35rem 0.25rem 1rem;
        margin-bottom: 0.45rem;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        margin-bottom: 0.75rem;
        border-radius: 12px;
        background: var(--primary);
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.2);
    }

    .sidebar-title {
        margin: 0;
        color: var(--navy);
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .sidebar-subtitle {
        margin-top: 0.25rem;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }

    div[role="radiogroup"] label {
        padding: 0.55rem 0.6rem;
        margin-bottom: 0.2rem;
        border-radius: 9px;
    }

    div[role="radiogroup"] label:hover {
        background: var(--primary-soft);
    }

    .hero {
        padding: 2rem 2.15rem;
        margin-bottom: 1.15rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 10px 35px rgba(15, 23, 42, 0.05);
    }

    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin-bottom: 0.8rem;
        color: var(--primary);
        background: var(--primary-soft);
        border: 1px solid #DBEAFE;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;
        color: var(--navy);
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 850;
        letter-spacing: -0.045em;
        line-height: 1.05;
    }

    .hero-description {
        max-width: 800px;
        margin-top: 0.8rem;
        margin-bottom: 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
    }

    .section-heading {
        margin-top: 1.5rem;
        margin-bottom: 0.85rem;
    }

    .section-heading h2 {
        margin: 0;
        color: var(--navy);
        font-size: 1.4rem;
        font-weight: 780;
        letter-spacing: -0.025em;
    }

    .section-heading p {
        margin: 0.3rem 0 0;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .card {
        height: 100%;
        padding: 1.1rem 1.15rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.035);
    }

    .card-label {
        margin-bottom: 0.4rem;
        color: var(--muted);
        font-size: 0.73rem;
        font-weight: 720;
        letter-spacing: 0.055em;
        text-transform: uppercase;
    }

    .card-value {
        margin: 0;
        color: var(--navy);
        font-size: 1.65rem;
        font-weight: 820;
        letter-spacing: -0.035em;
    }

    .card-detail {
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 0.79rem;
        line-height: 1.45;
    }

    .insight-card {
        padding: 1rem 1.05rem;
        margin-bottom: 0.75rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.03);
    }

    .insight-card h4 {
        margin: 0 0 0.35rem;
        color: var(--navy);
        font-size: 0.98rem;
    }

    .insight-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.6;
    }

    .prediction-card {
        padding: 1.4rem 1.45rem;
        margin-top: 0.75rem;
        background: linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 72%);
        border: 1px solid #BFDBFE;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
    }

    .prediction-card .label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .prediction-card .value {
        color: var(--primary-dark);
        font-size: 2.5rem;
        font-weight: 850;
        letter-spacing: -0.045em;
    }

    .prediction-card .detail {
        margin-top: 0.5rem;
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.55;
    }

    .positive-box,
    .negative-box,
    .neutral-box {
        padding: 0.9rem 1rem;
        margin-top: 0.75rem;
        border-radius: 11px;
        font-weight: 700;
    }

    .positive-box {
        color: var(--success);
        background: var(--success-soft);
        border: 1px solid #BBF7D0;
    }

    .negative-box {
        color: var(--danger);
        background: var(--danger-soft);
        border: 1px solid #FECACA;
    }

    .neutral-box {
        color: var(--primary-dark);
        background: var(--primary-soft);
        border: 1px solid #BFDBFE;
    }

    .disclaimer {
        padding: 0.85rem 1rem;
        background: #F8FAFC;
        color: var(--muted);
        border: 1px solid var(--border);
        border-radius: 11px;
        font-size: 0.8rem;
        line-height: 1.55;
    }

    .status-pill {
        display: inline-block;
        padding: 0.32rem 0.65rem;
        color: var(--success);
        background: var(--success-soft);
        border: 1px solid #BBF7D0;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 720;
    }

    div[data-testid="stMetric"] {
        padding: 1rem 1.05rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.03);
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.8rem;
        background: var(--primary);
        color: #FFFFFF;
        border: 1px solid var(--primary);
        border-radius: 9px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--primary-dark);
        border-color: var(--primary-dark);
        color: #FFFFFF;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stPlotlyChart"] {
        overflow: hidden;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    .confidence-high,
    .confidence-medium,
    .confidence-low {
        padding: 0.85rem 1rem;
        margin-top: 0.75rem;
        border-radius: 11px;
        font-size: 0.86rem;
        font-weight: 700;
        line-height: 1.5;
    }

    .confidence-high {
        color: #15803D;
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
    }

    .confidence-medium {
        color: #B45309;
        background: #FFFBEB;
        border: 1px solid #FDE68A;
    }

    .confidence-low {
        color: #B91C1C;
        background: #FEF2F2;
        border: 1px solid #FECACA;
    }


    .footer {
        margin-top: 2.5rem;
        padding-top: 1.25rem;
        color: var(--muted);
        border-top: 1px solid var(--border);
        font-size: 0.78rem;
        text-align: center;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.45rem;
        }

        .hero-title {
            font-size: 2.15rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Cached Loaders
# ============================================================

@st.cache_resource
def load_model() -> Any:
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata() -> dict[str, Any]:
    metadata = joblib.load(MODEL_METADATA_PATH)

    if not isinstance(metadata, dict):
        raise TypeError(
            "final_model_metadata.pkl must contain a dictionary."
        )

    return metadata


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    return value if isinstance(value, dict) else {}


# ============================================================
# Utility Functions
# ============================================================

def safe_float(
    value: Any,
    default: float = np.nan,
) -> float:
    try:
        numeric_value = float(value)

        if np.isfinite(numeric_value):
            return numeric_value

    except (TypeError, ValueError):
        pass

    return default


def format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    numeric_value = safe_float(value)

    if not np.isfinite(numeric_value):
        return "—"

    return f"{numeric_value:,.{decimals}f}"


def to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    return dataframe.to_csv(
        index=False
    ).encode("utf-8")


def render_hero(
    eyebrow: str,
    title: str,
    description: str,
) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-description">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(
    title: str,
    description: str,
) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    detail: str,
) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <p class="card-value">{value}</p>
            <div class="card-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(
    title: str,
    description: str,
) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(
    figure: go.Figure,
    height: int = 430,
    show_legend: bool = True,
) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(
            l=25,
            r=25,
            t=65,
            b=35,
        ),
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            color="#334155",
        ),
        title=dict(
            font=dict(
                color="#0F172A",
                size=18,
            ),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=show_legend,
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    figure.update_xaxes(
        showgrid=False,
        linecolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
    )

    figure.update_yaxes(
        gridcolor="#EEF2F7",
        linecolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
    )

    return figure


def predict_sales(
    input_values: dict[str, float],
) -> float:
    input_frame = pd.DataFrame(
        [
            {
                feature: float(
                    input_values[feature]
                )
                for feature in feature_columns
            }
        ]
    )

    input_frame = input_frame[
        feature_columns
    ]

    prediction = float(
        model.predict(
            input_frame
        )[0]
    )

    if not np.isfinite(prediction):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    return prediction


def validate_inputs(
    values: dict[str, float],
) -> list[str]:
    warnings: list[str] = []

    for feature, value in values.items():
        if value < 0:
            warnings.append(
                f"{feature} budget cannot be negative."
            )
            continue

        feature_range = input_ranges[
            feature
        ]

        if value < feature_range["minimum"]:
            warnings.append(
                f"{feature} is below the historical minimum "
                f"of {feature_range['minimum']:.2f}."
            )

        if value > feature_range["maximum"]:
            warnings.append(
                f"{feature} is above the historical maximum "
                f"of {feature_range['maximum']:.2f}. "
                "The prediction may be less reliable."
            )

    return warnings


def prediction_confidence(
    values: dict[str, float],
) -> tuple[str, str, str]:
    """
    Classify prediction confidence using historical feature ranges.

    Returns
    -------
    tuple
        Confidence label, CSS class, and explanation.
    """

    outside_count = 0
    severe_count = 0

    for feature, value in values.items():
        feature_range = input_ranges[feature]

        minimum = float(
            feature_range["minimum"]
        )

        maximum = float(
            feature_range["maximum"]
        )

        observed_width = max(
            maximum - minimum,
            1e-9,
        )

        if value < minimum:
            outside_count += 1

            if (
                minimum - value
            ) > 0.10 * observed_width:
                severe_count += 1

        elif value > maximum:
            outside_count += 1

            if (
                value - maximum
            ) > 0.10 * observed_width:
                severe_count += 1

    if outside_count == 0:
        return (
            "High Confidence",
            "confidence-high",
            (
                "All advertising inputs are within the historical "
                "training range."
            ),
        )

    if outside_count == 1 and severe_count == 0:
        return (
            "Medium Confidence",
            "confidence-medium",
            (
                "One input is slightly outside the historical range. "
                "Use the result as a directional estimate."
            ),
        )

    return (
        "Low Confidence",
        "confidence-low",
        (
            "Multiple inputs or large deviations fall outside the "
            "historical range. Treat the prediction cautiously."
        ),
    )


def build_business_interpretation(
    values: dict[str, float],
    predicted_sales: float,
    sales_change: float,
    sales_change_percentage: float,
    confidence_label: str,
) -> str:
    """
    Generate a concise business interpretation for one prediction.
    """

    largest_channel = max(
        values,
        key=values.get,
    )

    if sales_change > 0:
        direction_text = (
            f"The submitted mix is predicted to generate "
            f"<strong>{predicted_sales:.2f}</strong> sales units, "
            f"which is <strong>{sales_change_percentage:+.2f}%</strong> "
            "above the historical baseline."
        )

    elif sales_change < 0:
        direction_text = (
            f"The submitted mix is predicted to generate "
            f"<strong>{predicted_sales:.2f}</strong> sales units, "
            f"which is <strong>{sales_change_percentage:+.2f}%</strong> "
            "below the historical baseline."
        )

    else:
        direction_text = (
            f"The submitted mix is predicted to generate "
            f"<strong>{predicted_sales:.2f}</strong> sales units, "
            "which matches the historical baseline."
        )

    return (
        f"{direction_text} "
        f"<strong>{largest_channel}</strong> receives the largest "
        "share of the entered budget. "
        f"Prediction confidence is <strong>{confidence_label}</strong>. "
        "This estimate should support planning, not replace controlled "
        "campaign testing."
    )


def create_sales_gauge(
    predicted_sales: float,
    historical_minimum: float,
    historical_maximum: float,
) -> go.Figure:
    """
    Create a professional Plotly gauge for predicted sales.
    """

    axis_maximum = max(
        historical_maximum * 1.15,
        predicted_sales * 1.10,
        1.0,
    )

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=predicted_sales,
            number={
                "valueformat": ".2f",
                "font": {
                    "color": "#0F172A",
                    "size": 42,
                },
            },
            delta={
                "reference": baseline_prediction,
                "valueformat": ".2f",
                "increasing": {
                    "color": "#15803D",
                },
                "decreasing": {
                    "color": "#B91C1C",
                },
            },
            title={
                "text": "Expected Sales",
                "font": {
                    "color": "#64748B",
                    "size": 17,
                },
            },
            gauge={
                "axis": {
                    "range": [0, axis_maximum],
                    "tickcolor": "#94A3B8",
                },
                "bar": {
                    "color": "#2563EB",
                    "thickness": 0.32,
                },
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [
                            0,
                            historical_minimum,
                        ],
                        "color": "#F8FAFC",
                    },
                    {
                        "range": [
                            historical_minimum,
                            historical_maximum,
                        ],
                        "color": "#EFF6FF",
                    },
                    {
                        "range": [
                            historical_maximum,
                            axis_maximum,
                        ],
                        "color": "#FFF7ED",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": "#0F172A",
                        "width": 3,
                    },
                    "thickness": 0.75,
                    "value": baseline_prediction,
                },
            },
        )
    )

    figure.update_layout(
        height=360,
        margin=dict(
            l=35,
            r=35,
            t=65,
            b=20,
        ),
        paper_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, Segoe UI, sans-serif",
        ),
    )

    return figure



def reliability_details(
    values: dict[str, float],
) -> list[str]:
    """Return transparent reasons for the reliability classification."""

    details: list[str] = []

    for feature, value in values.items():
        feature_range = input_ranges[feature]
        minimum = float(feature_range["minimum"])
        maximum = float(feature_range["maximum"])

        if minimum <= value <= maximum:
            details.append(
                f"✓ {feature} is within its historical range "
                f"({minimum:.2f}–{maximum:.2f})."
            )
        elif value < minimum:
            details.append(
                f"⚠ {feature} is below its historical minimum "
                f"of {minimum:.2f}."
            )
        else:
            details.append(
                f"⚠ {feature} is above its historical maximum "
                f"of {maximum:.2f}."
            )

    return details


def historical_percentile(
    feature: str,
    value: float,
) -> float:
    """Return the empirical percentile of an input within historical data."""

    return float(
        (data[feature] <= value).mean()
        * 100
    )


def position_in_historical_range(
    feature: str,
    value: float,
) -> float:
    """Return position within the observed min-max range, clipped to 0–100%."""

    minimum = float(input_ranges[feature]["minimum"])
    maximum = float(input_ranges[feature]["maximum"])

    if maximum <= minimum:
        return 0.0

    position = (
        (value - minimum)
        / (maximum - minimum)
        * 100
    )

    return float(
        np.clip(position, 0.0, 100.0)
    )


def build_preset_scenarios() -> dict[str, dict[str, float]]:
    """Build reusable demonstration scenarios from historical quantiles."""

    return {
        "Historical Average": {
            feature: float(data[feature].mean())
            for feature in feature_columns
        },
        "Low Budget": {
            feature: float(data[feature].quantile(0.25))
            for feature in feature_columns
        },
        "Balanced": {
            feature: float(data[feature].median())
            for feature in feature_columns
        },
        "TV Focused": {
            "TV": float(data["TV"].quantile(0.80)),
            "Radio": float(data["Radio"].median()),
            "Newspaper": float(data["Newspaper"].quantile(0.25)),
        },
        "Radio Focused": {
            "TV": float(data["TV"].median()),
            "Radio": float(data["Radio"].quantile(0.80)),
            "Newspaper": float(data["Newspaper"].quantile(0.25)),
        },
        "High Investment": {
            feature: float(data[feature].quantile(0.90))
            for feature in feature_columns
        },
    }


def build_alternative_scenarios(
    current_values: dict[str, float],
    current_prediction: float,
) -> pd.DataFrame:
    """Compare the current input with a 10% increase in each channel."""

    records = [
        {
            "Scenario": "Current Mix",
            **current_values,
            "Predicted Sales": current_prediction,
            "Change vs Current": 0.0,
        }
    ]

    for channel in feature_columns:
        alternative_values = current_values.copy()
        alternative_values[channel] = (
            alternative_values[channel]
            * 1.10
        )

        alternative_prediction = predict_sales(
            alternative_values
        )

        records.append(
            {
                "Scenario": f"{channel} +10%",
                **alternative_values,
                "Predicted Sales": alternative_prediction,
                "Change vs Current": (
                    alternative_prediction
                    - current_prediction
                ),
            }
        )

    return pd.DataFrame(records)


def evidence_based_recommendation(
    alternative_scenarios: pd.DataFrame,
    confidence_label: str,
) -> str:
    """Generate a recommendation grounded in scenario comparisons."""

    alternatives = alternative_scenarios[
        alternative_scenarios["Scenario"] != "Current Mix"
    ].sort_values(
        "Change vs Current",
        ascending=False,
    )

    best_row = alternatives.iloc[0]
    weakest_channel = str(
        channel_correlations.sort_values(
            "Correlation with Sales",
            ascending=True,
        ).iloc[0]["Channel"]
    )

    confidence_note = (
        "The current inputs remain within the historical range."
        if confidence_label == "High Confidence"
        else "Validate this direction with a controlled pilot because some inputs are outside the historical range."
    )

    return (
        f"Among the tested next-step options, <strong>{best_row['Scenario']}</strong> "
        f"produces the largest incremental prediction "
        f"(<strong>{best_row['Change vs Current']:+.2f}</strong> sales units). "
        f"{weakest_channel} has the weakest simple historical relationship with sales, "
        "so additional budget there should be reviewed carefully rather than assumed to be effective. "
        f"{confidence_note}"
    )



def build_budget_allocation_suggestions(
    total_budget: float,
) -> pd.DataFrame:
    """
    Compare a small set of transparent fixed-budget allocations.

    The result is model-guided, not a mathematical optimum.
    """

    if total_budget < 0:
        raise ValueError(
            "Total budget cannot be negative."
        )

    historical_totals = {
        feature: float(data[feature].sum())
        for feature in feature_columns
    }

    historical_sum = sum(
        historical_totals.values()
    )

    historical_shares = {
        feature: (
            historical_totals[feature]
            / historical_sum
            if historical_sum > 0
            else 1.0 / len(feature_columns)
        )
        for feature in feature_columns
    }

    allocation_templates = {
        "Historical Mix": historical_shares,
        "Balanced Mix": {
            feature: 1.0 / len(feature_columns)
            for feature in feature_columns
        },
        "TV Focused": {
            "TV": 0.60,
            "Radio": 0.30,
            "Newspaper": 0.10,
        },
        "Radio Focused": {
            "TV": 0.45,
            "Radio": 0.45,
            "Newspaper": 0.10,
        },
        "TV + Radio Focused": {
            "TV": 0.65,
            "Radio": 0.30,
            "Newspaper": 0.05,
        },
    }

    records: list[dict[str, float | str]] = []

    for scenario_name, shares in allocation_templates.items():
        allocation = {
            feature: float(
                total_budget
                * shares.get(
                    feature,
                    0.0,
                )
            )
            for feature in feature_columns
        }

        predicted_sales = predict_sales(
            allocation
        )

        records.append(
            {
                "Allocation": scenario_name,
                **allocation,
                "Total Budget": total_budget,
                "Predicted Sales": predicted_sales,
                "Change vs Baseline": (
                    predicted_sales
                    - baseline_prediction
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "Predicted Sales",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def feature_value_column(
    dataframe: pd.DataFrame,
) -> str | None:
    for column in [
        "Model Strength",
        "Importance",
        "Absolute Coefficient",
        "Coefficient",
    ]:
        if column in dataframe.columns:
            return column

    return None


# ============================================================
# Load Project Assets
# ============================================================

required_files = [
    DATA_PATH,
    MODEL_PATH,
    MODEL_METADATA_PATH,
]

missing_required_files = [
    path
    for path in required_files
    if not path.exists()
]

if missing_required_files:
    st.error(
        "AdIntel cannot start because required files are missing."
    )

    for missing_path in missing_required_files:
        st.code(str(missing_path))

    st.info(
        "Run Notebooks 1–6 before starting the Streamlit app."
    )
    st.stop()

try:
    model = load_model()
    metadata = load_metadata()

    data = load_csv(DATA_PATH)
    streamlit_config = load_json(
        STREAMLIT_CONFIG_PATH
    )
    business_summary = load_json(
        BUSINESS_SUMMARY_PATH
    )

    scenario_data = load_csv(
        SCENARIO_PATH
    )
    sensitivity_data = load_csv(
        SENSITIVITY_PATH
    )
    reallocation_data = load_csv(
        REALLOCATION_PATH
    )
    channel_insights = load_csv(
        CHANNEL_INSIGHTS_PATH
    )
    recommendations = load_csv(
        RECOMMENDATIONS_PATH
    )
    tuned_metrics = load_csv(
        MODEL_METRICS_PATH
    )
    cv_results = load_csv(
        CV_RESULTS_PATH
    )
    prediction_data = load_csv(
        FINAL_PREDICTIONS_PATH
    )
    feature_importance = load_csv(
        FEATURE_IMPORTANCE_PATH
    )
    baseline_tuned_data = load_csv(
        BASELINE_TUNED_PATH
    )

except Exception as error:
    st.error(
        "AdIntel encountered an error while loading project assets."
    )
    st.exception(error)
    st.stop()


# ============================================================
# Validate Configuration
# ============================================================

feature_columns = metadata.get(
    "feature_columns",
    ["TV", "Radio", "Newspaper"],
)

target_column = str(
    metadata.get(
        "target_column",
        "Sales",
    )
)

model_name = str(
    metadata.get(
        "model_name",
        type(model).__name__,
    )
)

required_columns = (
    feature_columns
    + [target_column]
)

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if data.empty or missing_columns:
    st.error(
        "The cleaned dataset is empty or missing required columns."
    )

    if missing_columns:
        st.code(str(missing_columns))

    st.stop()

test_r2 = safe_float(
    metadata.get("test_r2")
)
test_mae = safe_float(
    metadata.get("test_mae")
)
test_rmse = safe_float(
    metadata.get("test_rmse")
)
test_mape = safe_float(
    metadata.get("test_mape_percent")
)
cv_rmse = safe_float(
    metadata.get("cv_mean_rmse")
)

default_ranges = {
    feature: {
        "minimum": float(
            data[feature].min()
        ),
        "maximum": float(
            data[feature].max()
        ),
        "average": float(
            data[feature].mean()
        ),
        "median": float(
            data[feature].median()
        ),
    }
    for feature in feature_columns
}

saved_ranges = streamlit_config.get(
    "input_ranges",
    {}
)

input_ranges: dict[
    str,
    dict[str, float]
] = {}

for feature in feature_columns:
    feature_saved = (
        saved_ranges.get(
            feature,
            {},
        )
        if isinstance(
            saved_ranges,
            dict,
        )
        else {}
    )

    input_ranges[feature] = {
        key: safe_float(
            feature_saved.get(
                key,
            ),
            default_ranges[
                feature
            ][key],
        )
        for key in [
            "minimum",
            "maximum",
            "average",
            "median",
        ]
    }

default_baseline = {
    feature: float(
        data[feature].mean()
    )
    for feature in feature_columns
}

saved_baseline = streamlit_config.get(
    "baseline_scenario",
    {},
)

baseline_scenario = {
    feature: safe_float(
        saved_baseline.get(
            feature,
        )
        if isinstance(
            saved_baseline,
            dict,
        )
        else None,
        default_baseline[
            feature
        ],
    )
    for feature in feature_columns
}

baseline_prediction = safe_float(
    streamlit_config.get(
        "baseline_prediction"
    )
)

if not np.isfinite(
    baseline_prediction
):
    baseline_prediction = predict_sales(
        baseline_scenario
    )

disclaimer = str(
    streamlit_config.get(
        "disclaimer",
        (
            "Predictions reflect historical associations and "
            "should not be interpreted as guaranteed sales or "
            "advertising return on investment."
        ),
    )
)

channel_correlations = (
    data[
        feature_columns
        + [target_column]
    ]
    .corr(
        numeric_only=True
    )[target_column]
    .drop(
        target_column
    )
    .sort_values(
        ascending=False
    )
    .reset_index()
)

channel_correlations.columns = [
    "Channel",
    "Correlation with Sales",
]

strongest_channel = str(
    business_summary.get(
        "strongest_model_channel",
        business_summary.get(
            "strongest_correlation_channel",
            channel_correlations.iloc[
                0
            ]["Channel"],
        ),
    )
)

plotly_config = {
    "displayModeBar": False,
    "responsive": True,
}


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">A</div>
            <h1 class="sidebar-title">AdIntel</h1>
            <div class="sidebar-subtitle">
                Intelligent Sales Prediction<br>
                & Advertising Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Explorer",
            "Sales Forecast",
            "Scenario Analysis",
            "Business Insights",
            "Model Performance",
            "About AdIntel",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<span class="status-pill">Model ready</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Final model: {model_name}"
    )

    if np.isfinite(test_r2):
        st.caption(
            f"Test R²: {test_r2:.3f}"
        )


# ============================================================
# Dashboard
# ============================================================

if selected_page == "Dashboard":
    render_hero(
        "Advertising intelligence platform",
        "Turn advertising data into clearer decisions.",
        (
            "Explore historical performance, understand channel "
            "relationships, and evaluate advertising scenarios "
            "through a deployment-ready machine learning model."
        ),
    )

    dashboard_kpis = st.columns(
        4
    )

    with dashboard_kpis[0]:
        render_kpi_card(
            "Historical Records",
            f"{len(data):,}",
            "Advertising observations used in the analysis.",
        )

    with dashboard_kpis[1]:
        render_kpi_card(
            "Average Sales",
            format_number(
                data[
                    target_column
                ].mean()
            ),
            "Mean sales outcome across historical records.",
        )

    with dashboard_kpis[2]:
        render_kpi_card(
            "Final Test R²",
            format_number(
                test_r2,
                decimals=3,
            ),
            "Variance explained on the held-out test data.",
        )

    with dashboard_kpis[3]:
        render_kpi_card(
            "Strongest Channel",
            strongest_channel,
            "Highest model strength or observed relationship.",
        )

    render_section(
        "Executive Overview",
        (
            "A concise view of historical sales and advertising "
            "allocation."
        ),
    )

    overview_left, overview_right = st.columns(
        [1.15, 0.85]
    )

    with overview_left:
        sales_chart = px.histogram(
            data,
            x=target_column,
            nbins=20,
            marginal="box",
            title="Historical Sales Distribution",
        )

        sales_chart.update_traces(
            marker_color="#2563EB",
            opacity=0.86,
        )

        st.plotly_chart(
            style_chart(
                sales_chart,
                height=430,
                show_legend=False,
            ),
            use_container_width=True,
            config=plotly_config,
        )

    with overview_right:
        budget_data = pd.DataFrame(
            {
                "Channel": feature_columns,
                "Total Budget": [
                    float(
                        data[
                            feature
                        ].sum()
                    )
                    for feature in feature_columns
                ],
            }
        )

        budget_chart = px.pie(
            budget_data,
            names="Channel",
            values="Total Budget",
            hole=0.62,
            title="Historical Budget Composition",
            color_discrete_sequence=[
                "#2563EB",
                "#60A5FA",
                "#BFDBFE",
            ],
        )

        budget_chart.update_traces(
            textposition="inside",
            textinfo="label+percent",
        )

        st.plotly_chart(
            style_chart(
                budget_chart,
                height=430,
            ),
            use_container_width=True,
            config=plotly_config,
        )

    render_section(
        "Advertising Relationship Overview",
        (
            "Compare the observed linear relationship between "
            "each advertising channel and sales."
        ),
    )

    correlation_chart_data = (
        channel_correlations
        .sort_values(
            "Correlation with Sales",
            ascending=True,
        )
    )

    correlation_chart = px.bar(
        correlation_chart_data,
        x="Correlation with Sales",
        y="Channel",
        orientation="h",
        text_auto=".3f",
        title="Observed Channel Correlation with Sales",
    )

    correlation_chart.update_traces(
        marker_color="#2563EB",
        textposition="outside",
    )

    st.plotly_chart(
        style_chart(
            correlation_chart,
            height=360,
            show_legend=False,
        ),
        use_container_width=True,
        config=plotly_config,
    )

    with st.expander(
        "View Interactive Correlation Heatmap"
    ):
        heatmap_columns = (
            feature_columns
            + [target_column]
        )

        correlation_matrix = data[
            heatmap_columns
        ].corr(
            numeric_only=True
        )

        heatmap = px.imshow(
            correlation_matrix,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Interactive Correlation Matrix",
        )

        heatmap.update_coloraxes(
            colorbar_title="Correlation"
        )

        st.plotly_chart(
            style_chart(
                heatmap,
                height=500,
                show_legend=False,
            ),
            use_container_width=True,
            config=plotly_config,
        )

        st.caption(
            "Correlation describes historical linear association. "
            "It does not establish causation or financial return."
        )

    best_scenario = business_summary.get(
        "best_increase_scenario",
        {},
    )

    best_reallocation = business_summary.get(
        "best_reallocation_scenario",
        {},
    )

    best_scenario_name = str(
        best_scenario.get(
            "scenario",
            "Scenario analysis available",
        )
        if isinstance(
            best_scenario,
            dict,
        )
        else "Scenario analysis available"
    )

    best_reallocation_name = str(
        best_reallocation.get(
            "reallocation",
            "Reallocation analysis available",
        )
        if isinstance(
            best_reallocation,
            dict,
        )
        else "Reallocation analysis available"
    )

    render_section(
        "Decision-Ready Highlights",
        (
            "Key findings generated from the trained model and "
            "scenario analysis."
        ),
    )

    highlight_columns = st.columns(
        3
    )

    with highlight_columns[0]:
        render_insight_card(
            "Priority Channel",
            (
                f"<strong>{strongest_channel}</strong> has the "
                "strongest model-based or observed relationship."
            ),
        )

    with highlight_columns[1]:
        render_insight_card(
            "Best Tested Increase",
            (
                f"<strong>{best_scenario_name}</strong> produced "
                "the strongest predicted improvement among the "
                "predefined scenarios."
            ),
        )

    with highlight_columns[2]:
        render_insight_card(
            "Best Fixed-Budget Move",
            (
                f"<strong>{best_reallocation_name}</strong> was "
                "the strongest tested equal-budget reallocation."
            ),
        )

# ============================================================
# Data Explorer
# ============================================================

elif selected_page == "Data Explorer":
    render_hero(
        "Interactive data exploration",
        "Explore the historical advertising dataset.",
        (
            "Filter advertising budgets and sales, review summary "
            "statistics, inspect relationships, and download the "
            "selected records."
        ),
    )

    render_section(
        "Filter Historical Records",
        (
            "Use the controls below to focus on a specific part of "
            "the observed dataset."
        ),
    )

    filter_columns = st.columns(
        2
    )

    selected_ranges: dict[
        str,
        tuple[float, float]
    ] = {}

    filter_features = (
        feature_columns
        + [target_column]
    )

    for index, feature in enumerate(
        filter_features
    ):
        minimum_value = float(
            data[feature].min()
        )

        maximum_value = float(
            data[feature].max()
        )

        with filter_columns[
            index % 2
        ]:
            selected_ranges[
                feature
            ] = st.slider(
                f"{feature} Range",
                min_value=minimum_value,
                max_value=maximum_value,
                value=(
                    minimum_value,
                    maximum_value,
                ),
                step=0.1,
                key=f"explorer_{feature}",
            )

    filtered_data = data.copy()

    for feature, (
        minimum_value,
        maximum_value,
    ) in selected_ranges.items():
        filtered_data = filtered_data[
            filtered_data[
                feature
            ].between(
                minimum_value,
                maximum_value,
                inclusive="both",
            )
        ]

    explorer_kpis = st.columns(
        4
    )

    with explorer_kpis[0]:
        render_kpi_card(
            "Filtered Records",
            f"{len(filtered_data):,}",
            f"{len(filtered_data) / len(data) * 100:.1f}% of the dataset.",
        )

    with explorer_kpis[1]:
        render_kpi_card(
            "Average Sales",
            format_number(
                filtered_data[
                    target_column
                ].mean()
                if not filtered_data.empty
                else np.nan
            ),
            "Mean sales within the selected filters.",
        )

    with explorer_kpis[2]:
        render_kpi_card(
            "Median Sales",
            format_number(
                filtered_data[
                    target_column
                ].median()
                if not filtered_data.empty
                else np.nan
            ),
            "Median sales within the selected filters.",
        )

    with explorer_kpis[3]:
        render_kpi_card(
            "Average Total Budget",
            format_number(
                filtered_data[
                    feature_columns
                ].sum(
                    axis=1
                ).mean()
                if not filtered_data.empty
                else np.nan
            ),
            "Mean combined advertising budget.",
        )

    if filtered_data.empty:
        st.warning(
            "No records match the selected filters. "
            "Widen one or more ranges."
        )
    else:
        render_section(
            "Filtered Data",
            (
                "Inspect the selected historical observations and "
                "download them for further analysis."
            ),
        )

        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Filtered Data",
            data=to_csv_bytes(
                filtered_data
            ),
            file_name=(
                "adintel_filtered_data.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        render_section(
            "Relationship Explorer",
            (
                "Select an advertising channel to inspect its "
                "relationship with sales."
            ),
        )

        selected_feature = st.selectbox(
            "Advertising Channel",
            feature_columns,
            key="explorer_feature",
        )

        relationship_chart = px.scatter(
            filtered_data,
            x=selected_feature,
            y=target_column,
            title=(
                f"{selected_feature} Budget vs "
                f"{target_column}"
            ),
            hover_data=[
                feature
                for feature in feature_columns
                if feature != selected_feature
            ],
        )

        relationship_chart.update_traces(
            marker=dict(
                color="#2563EB",
                size=8,
                opacity=0.72,
            ),
            selector=dict(
                mode="markers"
            ),
        )

        st.plotly_chart(
            style_chart(
                relationship_chart,
                height=460,
                show_legend=False,
            ),
            use_container_width=True,
            config=plotly_config,
        )

        render_section(
            "Filtered Summary Statistics",
            (
                "Review descriptive statistics for the currently "
                "selected observations."
            ),
        )

        summary_statistics = (
            filtered_data[
                filter_features
            ]
            .describe()
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Variable",
                }
            )
        )

        st.dataframe(
            summary_statistics.style.format(
                {
                    column: "{:.2f}"
                    for column in summary_statistics.columns
                    if column != "Variable"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Sales Forecast
# ============================================================

elif selected_page == "Sales Forecast":
    render_hero(
        "Interactive prediction",
        "Forecast sales and compare practical next steps.",
        (
            "Use historical presets or custom budgets to generate "
            "a prediction, review its approximate error range, "
            "understand reliability, and compare channel alternatives."
        ),
    )

    preset_scenarios = build_preset_scenarios()

    for feature in feature_columns:
        state_key = f"forecast_{feature}"

        if state_key not in st.session_state:
            st.session_state[state_key] = float(
                baseline_scenario[feature]
            )

    render_section(
        "Preset Scenarios",
        (
            "Apply a realistic historical scenario for a faster "
            "demonstration, or enter custom values below."
        ),
    )

    preset_column, apply_column, reset_column = st.columns(
        [1.7, 0.65, 0.8]
    )

    with preset_column:
        selected_preset = st.selectbox(
            "Scenario Preset",
            list(preset_scenarios.keys()),
            help=(
                "Presets are generated from historical averages, "
                "medians, and quantiles."
            ),
        )

    with apply_column:
        st.write("")
        st.write("")

        if st.button(
            "Apply Preset",
            use_container_width=True,
        ):
            for feature in feature_columns:
                st.session_state[
                    f"forecast_{feature}"
                ] = float(
                    preset_scenarios[
                        selected_preset
                    ][feature]
                )

            st.rerun()

    with reset_column:
        st.write("")
        st.write("")

        if st.button(
            "Reset to Average",
            use_container_width=True,
        ):
            for feature in feature_columns:
                st.session_state[
                    f"forecast_{feature}"
                ] = float(
                    baseline_scenario[feature]
                )

            st.rerun()

    render_section(
        "Advertising Inputs",
        (
            "Values within historical ranges provide the most "
            "credible model-based estimates."
        ),
    )

    input_columns = st.columns(
        len(feature_columns)
    )

    user_inputs: dict[str, float] = {}

    for column, feature in zip(
        input_columns,
        feature_columns,
    ):
        current_range = input_ranges[feature]
        state_key = f"forecast_{feature}"

        with column:
            user_inputs[feature] = st.number_input(
                f"{feature} Budget",
                min_value=0.0,
                step=1.0,
                key=state_key,
                help=(
                    "Historical range: "
                    f"{current_range['minimum']:.2f} to "
                    f"{current_range['maximum']:.2f}"
                ),
            )

            percentile = historical_percentile(
                feature,
                user_inputs[feature],
            )

            range_position = position_in_historical_range(
                feature,
                user_inputs[feature],
            )

            st.progress(
                int(round(range_position))
            )

            st.caption(
                f"Historical percentile: {percentile:.0f}th · "
                f"Range position: {range_position:.0f}%"
            )

    warning_messages = validate_inputs(
        user_inputs
    )

    for message in warning_messages:
        st.warning(
            f"{message} The estimate may be less reliable because "
            "the model has limited evidence in this region."
        )

    try:
        predicted_sales = predict_sales(
            user_inputs
        )
    except Exception as error:
        st.error(
            "The prediction could not be generated."
        )
        st.exception(error)
        st.stop()

    total_budget = float(
        sum(user_inputs.values())
    )

    sales_change = (
        predicted_sales
        - baseline_prediction
    )

    sales_change_percentage = (
        sales_change
        / baseline_prediction
        * 100
        if baseline_prediction != 0
        else np.nan
    )

    lower_bound = (
        max(0.0, predicted_sales - test_rmse)
        if np.isfinite(test_rmse)
        else np.nan
    )

    upper_bound = (
        predicted_sales + test_rmse
        if np.isfinite(test_rmse)
        else np.nan
    )

    (
        confidence_label,
        confidence_class,
        confidence_message,
    ) = prediction_confidence(
        user_inputs
    )

    confidence_reasons = reliability_details(
        user_inputs
    )

    result_left, result_right = st.columns(
        [1.05, 0.95]
    )

    with result_left:
        expected_range_html = (
            f"<br>Approximate range: <strong>{lower_bound:.2f}–{upper_bound:.2f}</strong> "
            "(prediction ± test RMSE)."
            if np.isfinite(lower_bound)
            and np.isfinite(upper_bound)
            else ""
        )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="label">Predicted Sales</div>
                <div class="value">{predicted_sales:.2f}</div>
                <div class="detail">
                    <strong>{sales_change_percentage:+.2f}%</strong>
                    compared with the historical baseline of
                    <strong>{baseline_prediction:.2f}</strong>.
                    {expected_range_html}
                    <br>Model: <strong>{model_name}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="{confidence_class}">
                {confidence_label}<br>
                <span style="font-weight:500;">
                    {confidence_message}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            "Why this reliability rating?"
        ):
            for reason in confidence_reasons:
                st.write(reason)

            st.caption(
                "Reliability is based on whether inputs remain within "
                "the feature ranges observed during training. It is not "
                "a statistical probability of correctness."
            )

    with result_right:
        st.plotly_chart(
            create_sales_gauge(
                predicted_sales=predicted_sales,
                historical_minimum=float(
                    data[target_column].min()
                ),
                historical_maximum=float(
                    data[target_column].max()
                ),
            ),
            use_container_width=True,
            config=plotly_config,
        )

    metric_columns = st.columns(4)

    with metric_columns[0]:
        st.metric(
            "Total Budget",
            f"{total_budget:.2f}",
        )

    with metric_columns[1]:
        st.metric(
            "Baseline Prediction",
            f"{baseline_prediction:.2f}",
        )

    with metric_columns[2]:
        st.metric(
            "Test RMSE",
            format_number(
                test_rmse,
                decimals=2,
            ),
        )

    with metric_columns[3]:
        delta_text = (
            f"{sales_change_percentage:+.2f}%"
            if np.isfinite(
                sales_change_percentage
            )
            else "Unavailable"
        )

        st.metric(
            "Change from Baseline",
            f"{sales_change:+.2f}",
            delta=delta_text,
        )

    if sales_change > 0:
        box_class = "positive-box"
        change_message = (
            f"This scenario is predicted to produce "
            f"{sales_change:.2f} more sales units than baseline."
        )
    elif sales_change < 0:
        box_class = "negative-box"
        change_message = (
            f"This scenario is predicted to produce "
            f"{abs(sales_change):.2f} fewer sales units than baseline."
        )
    else:
        box_class = "neutral-box"
        change_message = (
            "This scenario matches the baseline prediction."
        )

    st.markdown(
        f'<div class="{box_class}">{change_message}</div>',
        unsafe_allow_html=True,
    )

    alternative_scenarios = build_alternative_scenarios(
        current_values=user_inputs,
        current_prediction=predicted_sales,
    )

    render_section(
        "Recommended Next Test",
        (
            "Compare the current mix with a 10% increase in each "
            "channel to identify the strongest model-based next step."
        ),
    )

    recommendation_text = evidence_based_recommendation(
        alternative_scenarios=alternative_scenarios,
        confidence_label=confidence_label,
    )

    render_insight_card(
        "Evidence-Based Recommendation",
        recommendation_text,
    )

    alternative_chart = px.bar(
        alternative_scenarios,
        x="Scenario",
        y="Predicted Sales",
        text_auto=".2f",
        title="Current Mix vs 10% Channel Increases",
        color="Scenario",
        color_discrete_sequence=[
            "#94A3B8",
            "#2563EB",
            "#60A5FA",
            "#BFDBFE",
        ],
    )

    alternative_chart.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    st.plotly_chart(
        style_chart(
            alternative_chart,
            height=430,
            show_legend=False,
        ),
        use_container_width=True,
        config=plotly_config,
    )

    st.dataframe(
        alternative_scenarios.style.format(
            {
                column: "{:.2f}"
                for column in alternative_scenarios.columns
                if column != "Scenario"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    render_section(
        "Business Interpretation",
        (
            "Translate the result into a concise planning insight."
        ),
    )

    business_interpretation = build_business_interpretation(
        values=user_inputs,
        predicted_sales=predicted_sales,
        sales_change=sales_change,
        sales_change_percentage=sales_change_percentage,
        confidence_label=confidence_label,
    )

    render_insight_card(
        "Model-Based Insight",
        business_interpretation,
    )

    render_section(
        "Budget Breakdown",
        (
            "Review the submitted channel allocation and compare "
            "the prediction with baseline."
        ),
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        comparison_data = pd.DataFrame(
            {
                "Scenario": [
                    "Historical Baseline",
                    "Submitted Scenario",
                ],
                "Predicted Sales": [
                    baseline_prediction,
                    predicted_sales,
                ],
            }
        )

        comparison_chart = px.bar(
            comparison_data,
            x="Scenario",
            y="Predicted Sales",
            text_auto=".2f",
            title="Prediction Compared with Baseline",
            color="Scenario",
            color_discrete_map={
                "Historical Baseline": "#BFDBFE",
                "Submitted Scenario": "#2563EB",
            },
        )

        comparison_chart.update_traces(
            textposition="outside",
        )

        st.plotly_chart(
            style_chart(
                comparison_chart,
                height=390,
                show_legend=False,
            ),
            use_container_width=True,
            config=plotly_config,
        )

    with chart_right:
        allocation_data = pd.DataFrame(
            {
                "Channel": feature_columns,
                "Budget": [
                    user_inputs[feature]
                    for feature in feature_columns
                ],
            }
        )

        allocation_chart = px.pie(
            allocation_data,
            names="Channel",
            values="Budget",
            hole=0.62,
            title="Submitted Budget Allocation",
            color_discrete_sequence=[
                "#2563EB",
                "#60A5FA",
                "#BFDBFE",
            ],
        )

        allocation_chart.update_traces(
            textposition="inside",
            textinfo="label+percent",
        )

        st.plotly_chart(
            style_chart(
                allocation_chart,
                height=390,
            ),
            use_container_width=True,
            config=plotly_config,
        )

    budget_breakdown = allocation_data.copy()

    budget_breakdown["Share (%)"] = (
        budget_breakdown["Budget"]
        / total_budget
        * 100
        if total_budget > 0
        else 0.0
    )

    budget_breakdown["Historical Percentile"] = [
        historical_percentile(
            feature,
            user_inputs[feature],
        )
        for feature in feature_columns
    ]

    total_row = pd.DataFrame(
        [
            {
                "Channel": "Total",
                "Budget": total_budget,
                "Share (%)": (
                    100.0
                    if total_budget > 0
                    else 0.0
                ),
                "Historical Percentile": np.nan,
            }
        ]
    )

    budget_breakdown = pd.concat(
        [budget_breakdown, total_row],
        ignore_index=True,
    )

    st.dataframe(
        budget_breakdown.style.format(
            {
                "Budget": "{:.2f}",
                "Share (%)": "{:.2f}%",
                "Historical Percentile": "{:.0f}th",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    prediction_record = pd.DataFrame(
        [
            {
                **user_inputs,
                "Total Budget": total_budget,
                "Predicted Sales": predicted_sales,
                "Approximate Lower Bound": lower_bound,
                "Approximate Upper Bound": upper_bound,
                "Baseline Prediction": baseline_prediction,
                "Sales Change": sales_change,
                "Sales Change (%)": sales_change_percentage,
                "Reliability": confidence_label,
            }
        ]
    )

    download_columns = st.columns(2)

    with download_columns[0]:
        st.download_button(
            "Download Prediction",
            data=to_csv_bytes(prediction_record),
            file_name="adintel_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_columns[1]:
        st.download_button(
            "Download Alternative Scenarios",
            data=to_csv_bytes(alternative_scenarios),
            file_name="adintel_alternative_scenarios.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        f'<div class="disclaimer">{disclaimer} The displayed range is an approximate error band based on ± test RMSE, not a formal statistical confidence interval.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Scenario Analysis
# ============================================================

elif selected_page == "Scenario Analysis":
    render_hero(
        "What-if analysis",
        "Compare advertising strategies before acting.",
        (
            "Explore predefined scenarios, channel sensitivity, "
            "and equal-budget reallocations generated from the "
            "final machine learning model."
        ),
    )

    if scenario_data.empty:
        st.info(
            "Scenario files are unavailable. Run Notebook 6 "
            "to generate scenario-analysis assets."
        )
    else:
        render_section(
            "Predefined Scenario Comparison",
            (
                "Each scenario changes one advertising assumption "
                "and compares predicted sales with baseline."
            ),
        )

        scenario_chart = px.bar(
            scenario_data,
            x="Scenario",
            y="Predicted Sales",
            text_auto=".2f",
            title="Predicted Sales Across Advertising Scenarios",
        )

        scenario_colors = [
            "#BFDBFE"
            if str(
                scenario
            ).lower() == "baseline"
            else "#2563EB"
            for scenario in scenario_data[
                "Scenario"
            ]
        ]

        scenario_chart.update_traces(
            marker_color=scenario_colors,
            textposition="outside",
        )

        scenario_chart.update_xaxes(
            tickangle=-20,
        )

        st.plotly_chart(
            style_chart(
                scenario_chart,
                height=470,
                show_legend=False,
            ),
            use_container_width=True,
            config=plotly_config,
        )

        st.dataframe(
            scenario_data,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Scenario Comparison",
            data=to_csv_bytes(
                scenario_data
            ),
            file_name=(
                "adintel_scenario_comparison.csv"
            ),
            mime="text/csv",
        )

    if not sensitivity_data.empty:
        render_section(
            "Channel Sensitivity",
            (
                "Observe how model-predicted sales change when one "
                "channel increases and the others remain fixed."
            ),
        )

        sensitivity_chart = px.line(
            sensitivity_data,
            x="Budget Increase (%)",
            y="Sales Change",
            color="Channel",
            markers=True,
            title="Model-Predicted Sales Sensitivity by Channel",
        )

        sensitivity_chart.update_traces(
            line=dict(
                width=3
            ),
            marker=dict(
                size=8
            ),
        )

        sensitivity_chart.add_hline(
            y=0,
            line_dash="dash",
        )

        st.plotly_chart(
            style_chart(
                sensitivity_chart,
                height=450,
            ),
            use_container_width=True,
            config=plotly_config,
        )

    if not reallocation_data.empty:
        render_section(
            "Fixed-Budget Reallocation",
            (
                "Compare reallocations that preserve the same "
                "total advertising budget."
            ),
        )

        reallocation_chart_data = (
            reallocation_data
            .sort_values(
                "Sales Change",
                ascending=False,
            )
        )

        reallocation_chart = px.bar(
            reallocation_chart_data,
            x="Reallocation",
            y="Sales Change",
            text_auto="+.2f",
            title=(
                "Predicted Effect of Equal-Budget Reallocation"
            ),
        )

        reallocation_chart.update_traces(
            marker_color=[
                "#16A34A"
                if value > 0
                else "#DC2626"
                if value < 0
                else "#94A3B8"
                for value in reallocation_chart_data[
                    "Sales Change"
                ]
            ],
            textposition="outside",
        )

        reallocation_chart.add_hline(
            y=0,
            line_dash="dash",
        )

        reallocation_chart.update_xaxes(
            tickangle=-25,
        )

        st.plotly_chart(
            style_chart(
                reallocation_chart,
                height=480,
                show_legend=False,
            ),
            use_container_width=True,
            config=plotly_config,
        )

        best_row = (
            reallocation_chart_data
            .iloc[0]
        )

        render_insight_card(
            "Strongest Tested Reallocation",
            (
                f"<strong>{best_row['Reallocation']}</strong> "
                "generated a predicted sales change of "
                f"<strong>{safe_float(best_row['Sales Change'], 0.0):+.2f}</strong> "
                "while preserving total budget."
            ),
        )

    render_section(
        "Model-Guided Budget Suggestion",
        (
            "Enter a fixed total budget and compare several transparent "
            "allocation templates. This is a tested suggestion, not a "
            "mathematical or financial optimum."
        ),
    )

    suggestion_budget = st.number_input(
        "Total Budget for Allocation Comparison",
        min_value=0.0,
        value=float(
            sum(
                baseline_scenario.values()
            )
        ),
        step=10.0,
        key="suggestion_total_budget",
    )

    suggestion_data = build_budget_allocation_suggestions(
        suggestion_budget
    )

    best_suggestion = suggestion_data.iloc[0]

    suggestion_chart = px.bar(
        suggestion_data,
        x="Allocation",
        y="Predicted Sales",
        text_auto=".2f",
        title="Model-Guided Fixed-Budget Allocation Comparison",
        color="Allocation",
        color_discrete_sequence=[
            "#2563EB",
            "#60A5FA",
            "#93C5FD",
            "#BFDBFE",
            "#DBEAFE",
        ],
    )

    suggestion_chart.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    st.plotly_chart(
        style_chart(
            suggestion_chart,
            height=450,
            show_legend=False,
        ),
        use_container_width=True,
        config=plotly_config,
    )

    render_insight_card(
        "Strongest Tested Allocation",
        (
            f"For a total budget of <strong>{suggestion_budget:.2f}</strong>, "
            f"the strongest tested template is "
            f"<strong>{best_suggestion['Allocation']}</strong>, with "
            f"predicted sales of "
            f"<strong>{best_suggestion['Predicted Sales']:.2f}</strong>. "
            "Validate this direction with a controlled campaign before "
            "making a real spending decision."
        ),
    )

    st.dataframe(
        suggestion_data.style.format(
            {
                column: "{:.2f}"
                for column in suggestion_data.columns
                if column != "Allocation"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Allocation Comparison",
        data=to_csv_bytes(
            suggestion_data
        ),
        file_name=(
            "adintel_budget_allocation_suggestions.csv"
        ),
        mime="text/csv",
    )

    st.markdown(
        f'<div class="disclaimer">{disclaimer}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Business Insights
# ============================================================

elif selected_page == "Business Insights":
    render_hero(
        "Business interpretation",
        "Connect model results with practical decisions.",
        (
            "Review channel evidence, feature interpretation, "
            "and recommendations while keeping model limitations "
            "visible."
        ),
    )

    render_section(
        "Channel Evidence",
        (
            "Historical allocation, correlation, and model strength "
            "describe different aspects of the data."
        ),
    )

    interpretation_data = (
        channel_insights
        if not channel_insights.empty
        else feature_importance
    )

    if not interpretation_data.empty:
        st.dataframe(
            interpretation_data,
            use_container_width=True,
            hide_index=True,
        )

        value_column = feature_value_column(
            interpretation_data
        )

        feature_column = (
            "Feature"
            if "Feature"
            in interpretation_data.columns
            else "Channel"
            if "Channel"
            in interpretation_data.columns
            else None
        )

        if (
            value_column is not None
            and feature_column is not None
        ):
            feature_chart_data = (
                interpretation_data
                .copy()
            )

            sort_column = value_column

            if value_column == "Coefficient":
                feature_chart_data[
                    "Absolute Coefficient"
                ] = (
                    feature_chart_data[
                        "Coefficient"
                    ].abs()
                )

                sort_column = (
                    "Absolute Coefficient"
                )

            feature_chart_data = (
                feature_chart_data
                .sort_values(
                    sort_column,
                    ascending=True,
                )
            )

            feature_chart = px.bar(
                feature_chart_data,
                x=value_column,
                y=feature_column,
                orientation="h",
                text_auto=".4f",
                title="Model-Based Advertising Channel Strength",
            )

            feature_chart.update_traces(
                marker_color="#2563EB",
                textposition="outside",
            )

            st.plotly_chart(
                style_chart(
                    feature_chart,
                    height=360,
                    show_legend=False,
                ),
                use_container_width=True,
                config=plotly_config,
            )
    else:
        st.info(
            "Feature interpretation is unavailable."
        )

    render_section(
        "Recommendations",
        (
            "These recommendations summarize model evidence and "
            "should be tested through controlled campaigns."
        ),
    )

    if recommendations.empty:
        st.info(
            "Recommendation assets are unavailable. "
            "Run Notebook 6 to generate them."
        )
    else:
        for _, row in recommendations.iterrows():
            priority = str(
                row.get(
                    "Priority",
                    "Review",
                )
            )

            recommendation = str(
                row.get(
                    "Recommendation",
                    "Review the evidence.",
                )
            )

            evidence = str(
                row.get(
                    "Evidence",
                    "",
                )
            )

            caution = str(
                row.get(
                    "Caution",
                    "",
                )
            )

            render_insight_card(
                f"{priority} Priority",
                (
                    f"<strong>{recommendation}</strong><br><br>"
                    f"<strong>Evidence:</strong> {evidence}<br>"
                    f"<strong>Caution:</strong> {caution}"
                ),
            )

        st.download_button(
            "Download Recommendations",
            data=to_csv_bytes(
                recommendations
            ),
            file_name=(
                "adintel_business_recommendations.csv"
            ),
            mime="text/csv",
        )

    render_section(
        "Responsible Interpretation",
        (
            "AdIntel is a decision-support application, not an "
            "automated advertising-budget allocator."
        ),
    )

    limitation_columns = st.columns(
        3
    )

    with limitation_columns[0]:
        render_insight_card(
            "No Causal Guarantee",
            (
                "The model identifies historical associations. "
                "It does not prove that increasing one channel "
                "will directly cause higher sales."
            ),
        )

    with limitation_columns[1]:
        render_insight_card(
            "No ROI Calculation",
            (
                "The dataset does not include revenue, profit "
                "margins, campaign costs, or channel efficiency."
            ),
        )

    with limitation_columns[2]:
        render_insight_card(
            "Historical Range Matters",
            (
                "Predictions outside observed advertising ranges "
                "may be less reliable."
            ),
        )


# ============================================================
# Model Performance
# ============================================================

elif selected_page == "Model Performance":
    render_hero(
        "Model transparency",
        "Understand how the final model was selected.",
        (
            "Review test metrics, cross-validation performance, "
            "prediction quality, and residual diagnostics."
        ),
    )

    metric_columns = st.columns(
        4
    )

    metric_values = [
        (
            "Test R²",
            test_r2,
        ),
        (
            "Test MAE",
            test_mae,
        ),
        (
            "Test RMSE",
            test_rmse,
        ),
        (
            "CV RMSE",
            cv_rmse,
        ),
    ]

    for column, (
        label,
        value,
    ) in zip(
        metric_columns,
        metric_values,
    ):
        with column:
            st.metric(
                label,
                format_number(
                    value,
                    decimals=3,
                ),
            )

    test_records = (
        len(prediction_data)
        if not prediction_data.empty
        else int(round(len(data) * 0.20))
    )

    training_records = max(
        len(data) - test_records,
        0,
    )

    render_section(
        "Model Card",
        (
            "A compact technical summary of the deployment model "
            "and its evaluation setup."
        ),
    )

    model_card_columns = st.columns(4)

    model_card_items = [
        (
            "Final Model",
            model_name,
            "Selected using five-fold cross-validation RMSE.",
        ),
        (
            "Training Records",
            f"{training_records:,}",
            "Rows used for model fitting after the split.",
        ),
        (
            "Testing Records",
            f"{test_records:,}",
            "Held-out rows used for final evaluation.",
        ),
        (
            "Input Features",
            str(len(feature_columns)),
            ", ".join(feature_columns),
        ),
    ]

    for column, item in zip(
        model_card_columns,
        model_card_items,
    ):
        with column:
            render_kpi_card(*item)

    render_section(
        "Tuned Model Comparison",
        (
            "Candidate models were ranked primarily using "
            "cross-validation RMSE."
        ),
    )

    if not tuned_metrics.empty:
        metric_column = next(
            (
                column
                for column in [
                    "Best CV RMSE",
                    "CV Mean RMSE",
                    "Tuned CV RMSE",
                ]
                if column
                in tuned_metrics.columns
            ),
            None,
        )

        if (
            metric_column is not None
            and "Model"
            in tuned_metrics.columns
        ):
            model_chart_data = (
                tuned_metrics
                .sort_values(
                    metric_column,
                    ascending=True,
                )
            )

            model_chart = px.bar(
                model_chart_data,
                x="Model",
                y=metric_column,
                text_auto=".3f",
                title="Tuned Model Validation Comparison",
            )

            best_value = float(
                model_chart_data[
                    metric_column
                ].min()
            )

            model_chart.update_traces(
                marker_color=[
                    "#16A34A"
                    if np.isclose(
                        float(value),
                        best_value,
                    )
                    else "#2563EB"
                    for value in model_chart_data[
                        metric_column
                    ]
                ],
                textposition="outside",
            )

            model_chart.update_xaxes(
                tickangle=-20,
            )

            st.plotly_chart(
                style_chart(
                    model_chart,
                    height=460,
                    show_legend=False,
                ),
                use_container_width=True,
                config=plotly_config,
            )

        st.dataframe(
            tuned_metrics,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Tuned model metrics are unavailable."
        )

    if (
        not baseline_tuned_data.empty
        and {
            "Model",
            "Baseline CV RMSE",
            "Tuned CV RMSE",
        }.issubset(
            baseline_tuned_data.columns
        )
    ):
        render_section(
            "Baseline vs Tuned Performance",
            (
                "Confirm whether tuning improved validation error "
                "for each model family."
            ),
        )

        baseline_plot_data = baseline_tuned_data[
            [
                "Model",
                "Baseline CV RMSE",
                "Tuned CV RMSE",
            ]
        ].melt(
            id_vars="Model",
            var_name="Version",
            value_name="CV RMSE",
        )

        baseline_tuned_chart = px.bar(
            baseline_plot_data,
            x="Model",
            y="CV RMSE",
            color="Version",
            barmode="group",
            text_auto=".3f",
            title="Baseline vs Tuned Cross-Validation RMSE",
            color_discrete_map={
                "Baseline CV RMSE": "#BFDBFE",
                "Tuned CV RMSE": "#2563EB",
            },
        )

        baseline_tuned_chart.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        st.plotly_chart(
            style_chart(
                baseline_tuned_chart,
                height=450,
            ),
            use_container_width=True,
            config=plotly_config,
        )

    required_cv_columns = {
        "Model",
        "CV Mean RMSE",
        "CV RMSE Std",
    }

    if (
        not cv_results.empty
        and required_cv_columns.issubset(
            cv_results.columns
        )
    ):
        render_section(
            "Cross-Validation Stability",
            (
                "Compare average validation error with variability "
                "across folds."
            ),
        )

        stability_chart = px.scatter(
            cv_results,
            x="CV Mean RMSE",
            y="CV RMSE Std",
            text="Model",
            title="Cross-Validation Performance and Stability",
        )

        stability_chart.update_traces(
            marker=dict(
                color="#2563EB",
                size=12,
            ),
            textposition="top center",
        )

        st.plotly_chart(
            style_chart(
                stability_chart,
                height=450,
                show_legend=False,
            ),
            use_container_width=True,
            config=plotly_config,
        )

    required_prediction_columns = {
        "Actual Sales",
        "Predicted Sales",
    }

    if (
        not prediction_data.empty
        and required_prediction_columns.issubset(
            prediction_data.columns
        )
    ):
        render_section(
            "Prediction Diagnostics",
            (
                "Inspect prediction alignment, residual patterns, "
                "and test-observation error."
            ),
        )

        diagnostic_left, diagnostic_right = st.columns(
            2
        )

        with diagnostic_left:
            minimum_value = min(
                float(
                    prediction_data[
                        "Actual Sales"
                    ].min()
                ),
                float(
                    prediction_data[
                        "Predicted Sales"
                    ].min()
                ),
            )

            maximum_value = max(
                float(
                    prediction_data[
                        "Actual Sales"
                    ].max()
                ),
                float(
                    prediction_data[
                        "Predicted Sales"
                    ].max()
                ),
            )

            actual_chart = px.scatter(
                prediction_data,
                x="Actual Sales",
                y="Predicted Sales",
                title="Actual vs Predicted Sales",
            )

            actual_chart.update_traces(
                marker=dict(
                    color="#2563EB",
                    size=9,
                    opacity=0.78,
                )
            )

            actual_chart.add_shape(
                type="line",
                x0=minimum_value,
                y0=minimum_value,
                x1=maximum_value,
                y1=maximum_value,
                line=dict(
                    dash="dash",
                    color="#64748B",
                ),
            )

            st.plotly_chart(
                style_chart(
                    actual_chart,
                    height=440,
                    show_legend=False,
                ),
                use_container_width=True,
                config=plotly_config,
            )

        with diagnostic_right:
            if "Residual" in prediction_data.columns:
                residual_chart = px.scatter(
                    prediction_data,
                    x="Predicted Sales",
                    y="Residual",
                    title="Residuals vs Predicted Sales",
                )

                residual_chart.update_traces(
                    marker=dict(
                        color="#2563EB",
                        size=9,
                        opacity=0.78,
                    )
                )

                residual_chart.add_hline(
                    y=0,
                    line_dash="dash",
                )

                st.plotly_chart(
                    style_chart(
                        residual_chart,
                        height=440,
                        show_legend=False,
                    ),
                    use_container_width=True,
                    config=plotly_config,
                )

        if "Residual" in prediction_data.columns:
            residual_distribution = px.histogram(
                prediction_data,
                x="Residual",
                nbins=15,
                marginal="box",
                title="Residual Distribution",
            )

            residual_distribution.update_traces(
                marker_color="#2563EB"
            )

            residual_distribution.add_vline(
                x=0,
                line_dash="dash",
            )

            st.plotly_chart(
                style_chart(
                    residual_distribution,
                    height=420,
                    show_legend=False,
                ),
                use_container_width=True,
                config=plotly_config,
            )

        st.download_button(
            "Download Final Predictions",
            data=to_csv_bytes(
                prediction_data
            ),
            file_name=(
                "adintel_final_predictions.csv"
            ),
            mime="text/csv",
        )

    render_section(
        "Selection Methodology",
        (
            "The test set was reserved for final evaluation rather "
            "than repeatedly used for model selection."
        ),
    )

    render_insight_card(
        f"Final Model: {model_name}",
        (
            "The model was selected using five-fold "
            "cross-validation RMSE on the training data. "
            "The held-out test set was used only for final "
            "generalization evaluation."
        ),
    )

    if np.isfinite(test_mape):
        render_insight_card(
            "Additional Metric",
            (
                "The final model's test MAPE is "
                f"<strong>{test_mape:.2f}%</strong>. "
                "MAPE should be interpreted carefully when actual "
                "target values are close to zero."
            ),
        )


# ============================================================
# About AdIntel
# ============================================================

else:
    render_hero(
        "Portfolio case study",
        "About AdIntel.",
        (
            "AdIntel is an end-to-end machine learning and "
            "advertising analytics application designed to predict "
            "sales and support scenario-based marketing decisions."
        ),
    )

    render_section(
        "Project Objective",
        (
            "Move beyond a basic prediction notebook by combining "
            "machine learning, interactive analytics, model "
            "transparency, and business scenario analysis."
        ),
    )

    render_insight_card(
        "Business Problem",
        (
            "Businesses need to understand how advertising budgets "
            "across TV, Radio, and Newspaper relate to sales. "
            "AdIntel provides an interactive interface for exploring "
            "historical data, forecasting sales, comparing scenarios, "
            "and reviewing model-based recommendations."
        ),
    )

    render_section(
        "End-to-End Workflow",
        (
            "The project follows a complete and reproducible "
            "data-science lifecycle."
        ),
    )

    workflow_columns = st.columns(
        5
    )

    workflow_steps = [
        (
            "01",
            "Data Understanding",
        ),
        (
            "02",
            "Interactive EDA",
        ),
        (
            "03",
            "Preprocessing",
        ),
        (
            "04",
            "Model Comparison",
        ),
        (
            "05",
            "Tuning & Deployment",
        ),
    ]

    for column, (
        number,
        title,
    ) in zip(
        workflow_columns,
        workflow_steps,
    ):
        with column:
            render_kpi_card(
                number,
                title,
                "Documented in dedicated notebooks.",
            )

    render_section(
        "Technology Stack",
        (
            "Tools used to build the analysis, model, and application."
        ),
    )

    technology_stack = pd.DataFrame(
        {
            "Area": [
                "Programming",
                "Data Processing",
                "Visualization",
                "Machine Learning",
                "Deployment",
                "Model Persistence",
            ],
            "Technology": [
                "Python",
                "Pandas, NumPy",
                "Plotly",
                "Scikit-learn, XGBoost",
                "Streamlit",
                "Joblib",
            ],
        }
    )

    st.dataframe(
        technology_stack,
        use_container_width=True,
        hide_index=True,
    )

    render_section(
        "Key Capabilities",
        (
            "Features included to make AdIntel a stronger "
            "portfolio project."
        ),
    )

    capability_columns = st.columns(
        3
    )

    with capability_columns[0]:
        render_insight_card(
            "Interactive Forecasting",
            (
                "Generate sales predictions from custom "
                "advertising-budget inputs."
            ),
        )

    with capability_columns[1]:
        render_insight_card(
            "Scenario Analysis",
            (
                "Compare channel increases and fixed-budget "
                "reallocation strategies."
            ),
        )

    with capability_columns[2]:
        render_insight_card(
            "Model Transparency",
            (
                "Review validation metrics, residual behavior, "
                "and feature interpretation."
            ),
        )

    render_section(
        "Limitations",
        (
            "Professional analytics requires clearly communicating "
            "what the model cannot conclude."
        ),
    )

    st.markdown(
        """
        <div class="disclaimer">
            The dataset is small and includes only three advertising
            channels. It does not contain profit, seasonality,
            campaign quality, product price, customer segments, or
            economic conditions. AdIntel should therefore be used
            as a scenario-planning and learning tool—not as an
            automated real-world budget-allocation system.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        AdIntel · Intelligent Sales Prediction
        & Advertising Analytics
    </div>
    """,
    unsafe_allow_html=True,
)
import json
import os
import re
import time
from urllib.parse import unquote

import anthropic
import pandas as pd
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
RATE_LIMIT_SECONDS = 30
MAX_SAMPLE = 75
MODEL = "claude-haiku-4-5-20251001"
CSV_PATH = "Yelp Restaurant Reviews.csv"

# City suffixes that appear at the end of Yelp biz slugs
CITIES = [
    "las-vegas", "charlotte", "cleveland", "pittsburgh",
    "phoenix", "madison", "champaign", "scottsdale",
    "chandler", "lakewood", "urbana", "sidney",
]

RATING_FILTERS = {
    "All Ratings": None,
    "Positive (4–5 ★)": (4, 5),
    "Negative (1–2 ★)": (1, 2),
    "Neutral (3 ★)": (3, 3),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_business_name(url: str) -> str:
    """Convert a Yelp biz URL into a readable business name."""
    slug = unquote(url.rstrip("/").split("/biz/")[-1])
    slug = re.sub(r"-\d+$", "", slug)           # strip trailing location number
    for city in sorted(CITIES, key=len, reverse=True):   # longest match first
        if slug.endswith(f"-{city}"):
            slug = slug[: -(len(city) + 1)]
            break
    return slug.replace("-", " ").title()


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Review Text"])
    df["Review Text"] = df["Review Text"].astype(str)
    return df


@st.cache_data
def build_business_map(_df: pd.DataFrame) -> dict:
    """Return {readable_name: yelp_url} sorted alphabetically."""
    mapping: dict[str, str] = {}
    for url in sorted(_df["Yelp URL"].unique()):
        name = extract_business_name(url)
        # Resolve rare name collisions
        base, n = name, 2
        while base in mapping:
            base = f"{name} ({n})"
            n += 1
        mapping[base] = url
    return dict(sorted(mapping.items()))


def filter_reviews(df: pd.DataFrame, url: str, rating_range) -> pd.DataFrame:
    subset = df[df["Yelp URL"] == url]
    if rating_range:
        lo, hi = rating_range
        subset = subset[subset["Rating"].between(lo, hi)]
    return subset


def call_claude(reviews: list[str], business: str, filter_label: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        st.write("DEBUG: API key loaded from environment variable.")
    else:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
            st.write("DEBUG: API key loaded from st.secrets.")
        except Exception as e:
            st.write(f"DEBUG: Failed to load API key from st.secrets — {e}")
    st.write(f"DEBUG: api_key is {'set' if api_key else 'NOT SET'}")
    client = anthropic.Anthropic(api_key=api_key)
    numbered = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(reviews))
    prompt = f"""You are a consumer insights analyst. Below are {len(reviews)} Yelp reviews for \
"{business}" ({filter_label} filter).

Analyze these reviews and identify the top 5–7 recurring themes that matter most to customers.

Return ONLY valid JSON — no markdown fences, no explanation — using exactly this structure:
{{
  "themes": [
    {{
      "name": "Theme Name (2–5 words)",
      "description": "1–2 sentences summarizing what customers say about this theme.",
      "quotes": [
        "Short verbatim quote under 25 words",
        "Short verbatim quote under 25 words"
      ],
      "so_what": "One concrete, actionable strategic implication for the business."
    }}
  ]
}}

Reviews:
---
{numbered}
---"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if the model adds them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Rate limiter ──────────────────────────────────────────────────────────────

def seconds_until_allowed() -> int:
    last = st.session_state.get("last_call_ts", 0)
    remaining = RATE_LIMIT_SECONDS - (time.time() - last)
    return max(int(remaining), 0)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="John Cheek + Claude | AI Consumer Insights",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Metric row */
    [data-testid="metric-container"] {
        background: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }

    /* Quote styling */
    .quote-block {
        border-left: 3px solid #94a3b8;
        padding: 0.3rem 0.8rem;
        color: #475569;
        font-style: italic;
        margin: 0.35rem 0;
        font-size: 0.93rem;
        line-height: 1.5;
    }

    /* Strategic implication box */
    .so-what-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #166534;
    }

    /* Theme number badge */
    .theme-badge {
        display: inline-block;
        background: #3b82f6;
        color: white;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 10px;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }

    /* Expander header tweaks */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a3a5c 0%, #2563a8 55%, #1a3a5c 100%);
        border-radius: 12px;
        padding: 2.25rem 2rem 2rem;
        margin-bottom: 1.5rem;
    }
    .hero-stats {
        display: flex;
        gap: 1.25rem;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .stat-item {
        flex: 1;
        min-width: 140px;
        text-align: center;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 10px;
        padding: 1rem 0.75rem;
    }
    .stat-number {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
    }
    .stat-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.78);
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Contact bar */
    .contact-bar {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        flex-wrap: wrap;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.7rem 1.1rem;
        margin-top: 1rem;
        font-size: 0.91rem;
    }
    .contact-bar .contact-label {
        color: #64748b;
        font-weight: 500;
    }
    .contact-bar a {
        color: #2563a8;
        text-decoration: none;
        font-weight: 500;
    }
    .contact-bar a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("John Cheek + Claude")
st.caption("AI-Powered Consumer Insights")

# Hero banner with stat callouts
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-number">49</div>
                <div class="stat-label">Businesses Analyzed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">19,896</div>
                <div class="stat-label">Reviews</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">AI-Powered</div>
                <div class="stat-label">Theme Clustering</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Intro
st.markdown(
    """
Good research has always required two things: the right questions and the right tools. The questions haven't changed. The tools have.

This tool didn't build itself. I'm John Cheek, a consumer insights researcher with roots in audience strategy and advertising, and I came to this project with an idea: what if a researcher could use AI to do in minutes what once took weeks? I worked with Claude to ideate, design, and build something actually useful, not just technically impressive. That process, knowing what to build and why, is the part that still requires a human.

The demo analyzes real Yelp reviews across dessert and bakery concepts, extracting themes, representative consumer language, and strategic implications. The AI does the heavy lifting. The research thinking, what to ask, how to frame it, what it means for business, is still the hard part. That's the value I am ready to provide.
    """
)

# Contact bar
st.markdown(
    """
    <div class="contact-bar">
        <span class="contact-label">Get in touch:</span>
        <a href="mailto:jdcheek@gmail.com">jdcheek@gmail.com</a>
        <a href="https://www.linkedin.com/in/johncheek/" target="_blank">LinkedIn ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── How to use ────────────────────────────────────────────────────────────────
st.markdown(
    """
**How to use this tool:**

1. Select a restaurant from the dropdown to the left
2. Filter by review sentiment if desired — positive, neutral, negative, or all reviews
3. Click Analyze and see the results below
    """
)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
business_map = build_business_map(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Analysis Settings")

    selected_name = st.selectbox(
        "Business",
        options=list(business_map.keys()),
        help="49 businesses across Las Vegas, Charlotte, Cleveland, Pittsburgh, "
             "Phoenix, Madison, and Champaign.",
    )

    filter_label = st.radio(
        "Rating Filter",
        options=list(RATING_FILTERS.keys()),
        help="Filter the reviews sent to Claude by star rating.",
    )

    st.divider()

    wait = seconds_until_allowed()
    btn_disabled = wait > 0

    if btn_disabled:
        st.warning(f"⏳ Rate limit active — wait **{wait}s** before next request.")

    analyze = st.button(
        "🚀 Analyze Reviews",
        disabled=btn_disabled,
        use_container_width=True,
        type="primary",
    )
    st.caption(
        f"Sends up to **{MAX_SAMPLE}** randomly sampled reviews to Claude per run.  \n"
        f"Rate limit: 1 request per {RATE_LIMIT_SECONDS} seconds."
    )

# ── Stats row ─────────────────────────────────────────────────────────────────
selected_url = business_map[selected_name]
rating_range = RATING_FILTERS[filter_label]
subset = filter_reviews(df, selected_url, rating_range)
total_for_biz = len(df[df["Yelp URL"] == selected_url])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reviews on File", total_for_biz)
c2.metric("Matching Rating Filter", len(subset))
c3.metric("Reviews Sampled", min(len(subset), MAX_SAMPLE))
if len(subset) > 0:
    avg = float(subset["Rating"].mean())
    c4.metric("Avg Rating (filtered)", f"{avg:.2f} ★")
else:
    c4.metric("Avg Rating (filtered)", "—")

# ── Run analysis ──────────────────────────────────────────────────────────────
if analyze:
    if len(subset) == 0:
        st.error(
            "No reviews match this filter. Try selecting a different rating range."
        )
    else:
        sample_texts = (
            subset["Review Text"]
            .sample(min(len(subset), MAX_SAMPLE))
            .tolist()
        )
        st.session_state["last_call_ts"] = time.time()

        with st.spinner(f"Analyzing {len(sample_texts)} reviews with Claude AI…"):
            try:
                result = call_claude(sample_texts, selected_name, filter_label)
                st.session_state.update(
                    {
                        "result": result,
                        "result_business": selected_name,
                        "result_filter": filter_label,
                        "result_n": len(sample_texts),
                    }
                )
            except json.JSONDecodeError as exc:
                st.error(
                    f"Claude returned malformed JSON — please try again. ({exc})"
                )
                st.stop()
            except anthropic.APIError as exc:
                st.error(f"Anthropic API error: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

# ── Display results ───────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    biz = st.session_state["result_business"]
    filt = st.session_state["result_filter"]
    n = st.session_state["result_n"]

    st.subheader(f"Theme Analysis — {biz}")
    st.caption(f"{filt} · {n} reviews sampled · Powered by {MODEL}")

    themes = result.get("themes", [])

    if not themes:
        st.warning("No themes were returned. Try running the analysis again.")
    else:
        for i, theme in enumerate(themes, start=1):
            with st.expander(f"Theme {i}: {theme.get('name', 'Unnamed')}", expanded=True):
                left, right = st.columns([3, 2], gap="large")

                with left:
                    st.markdown(
                        f'<span class="theme-badge">Theme {i}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{theme.get('name', '')}**")
                    st.write(theme.get("description", ""))

                    quotes = theme.get("quotes", [])
                    if quotes:
                        st.markdown("**Voices from the reviews:**")
                        for q in quotes:
                            st.markdown(
                                f'<div class="quote-block">"{q}"</div>',
                                unsafe_allow_html=True,
                            )

                with right:
                    st.markdown("**💡 Strategic Implication**")
                    st.markdown(
                        f'<div class="so-what-box">{theme.get("so_what", "")}</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.caption(
            "⚠️ AI-generated analysis based on a random sample of reviews. "
            "Treat findings as directional hypotheses — validate with additional "
            "research before acting on them."
        )

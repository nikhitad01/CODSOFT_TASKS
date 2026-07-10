import os
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="CineRate AI",
    page_icon="🎬",
    layout="wide",
)


# ==================================================
# Light theme styling
# ==================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #111827;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div.stButton > button {
        width: 100%;
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
        border: none;
    }

    div.stDownloadButton > button {
        background-color: #FFFFFF;
        color: #1D4ED8;
        border: 1px solid #2563EB;
        border-radius: 10px;
        font-weight: 700;
    }

    div.stDownloadButton > button:hover {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #1D4ED8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Paths
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "imdb_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "movie_rating_xgb_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")
REC_MOVIES_PATH = os.path.join(BASE_DIR, "models", "recommendation_movies.csv")
COSINE_PATH = os.path.join(BASE_DIR, "models", "cosine_similarity.pkl")
MOVIE_INDICES_PATH = os.path.join(BASE_DIR, "models", "movie_indices.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_importance_xgb.csv",
)


# ==================================================
# Load data and model objects
# ==================================================
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_data = pd.read_csv(DATA_PATH)
    recommendation_data = pd.read_csv(REC_MOVIES_PATH)
    return feature_data, recommendation_data


@st.cache_resource
def load_model_objects():
    rating_model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    similarity_matrix = joblib.load(COSINE_PATH)
    indices = joblib.load(MOVIE_INDICES_PATH)

    return rating_model, feature_names, similarity_matrix, indices


try:
    df, rec_movies = load_data()
    model, model_features, cosine_sim, movie_indices = load_model_objects()

except FileNotFoundError as error:
    st.error(f"Required project file was not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ==================================================
# Utility functions
# ==================================================
def clean_display_name(value: Any) -> str:
    """Clean values for display without changing stored dataset values."""
    if pd.isna(value):
        return "N/A"

    return str(value).strip().lstrip("#").strip().title()


def safe_numeric(
    value: Any,
    fallback_series: pd.Series,
    as_integer: bool = False,
) -> float | int:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric_value):
        numeric_value = pd.to_numeric(
            fallback_series,
            errors="coerce",
        ).median()

    if pd.isna(numeric_value):
        numeric_value = 0

    return int(numeric_value) if as_integer else float(numeric_value)


def prepare_prediction_input(
    year: int,
    duration: int,
    votes: int,
    genre: str,
    director: str,
    actor_1: str,
) -> pd.DataFrame:
    # Important: initialize with 0.0 so float values can be assigned safely.
    input_df = pd.DataFrame(
        np.zeros((1, len(model_features)), dtype=float),
        columns=model_features,
    )

    current_year = 2026

    base_values = {
        "year": float(year),
        "duration": float(duration),
        "votes": float(votes),
        "movie_age": float(current_year - year),
        "genre_count": float(
            len([item for item in str(genre).split(",") if item.strip()])
        ),
        "log_votes": float(np.log1p(votes)),
        "duration_missing": 0.0,
        "votes_missing": 0.0,
        "year_missing": 0.0,
        "director_missing": 0.0,
        "actor_1_missing": 0.0,
    }

    for column, value in base_values.items():
        if column in input_df.columns:
            input_df.loc[0, column] = float(value)

    numeric_votes = pd.to_numeric(df["votes"], errors="coerce")
    vote_75 = numeric_votes.quantile(0.75)
    vote_90 = numeric_votes.quantile(0.90)

    if "popular_movie" in input_df.columns:
        input_df.loc[0, "popular_movie"] = float(votes >= vote_75)

    if "highly_popular_movie" in input_df.columns:
        input_df.loc[0, "highly_popular_movie"] = float(votes >= vote_90)

    if "vote_percentile" in input_df.columns:
        input_df.loc[0, "vote_percentile"] = float(
            (numeric_votes.dropna() < votes).mean()
        )

    if duration < 90:
        runtime_column = "runtime_Short"
    elif duration <= 150:
        runtime_column = "runtime_Medium"
    else:
        runtime_column = "runtime_Long"

    if runtime_column in input_df.columns:
        input_df.loc[0, runtime_column] = 1.0

    cleaned_genres = [
        item.strip().lower().replace(" ", "_")
        for item in str(genre).split(",")
        if item.strip()
    ]

    for cleaned_genre in cleaned_genres:
        genre_column = f"genre_{cleaned_genre}"
        if genre_column in input_df.columns:
            input_df.loc[0, genre_column] = 1.0

    main_genre = (
        cleaned_genres[0].replace("_", " ")
        if cleaned_genres
        else "unknown"
    )

    director_data = df[
        df["director"].astype(str).str.strip().str.lower()
        == str(director).strip().lower()
    ]

    actor_data = df[
        df["actor_1"].astype(str).str.strip().str.lower()
        == str(actor_1).strip().lower()
    ]

    overall_rating_mean = pd.to_numeric(
        df["rating"],
        errors="coerce",
    ).mean()

    director_rating_mean = pd.to_numeric(
        director_data["rating"],
        errors="coerce",
    ).mean()

    actor_rating_mean = pd.to_numeric(
        actor_data["rating"],
        errors="coerce",
    ).mean()

    if pd.isna(director_rating_mean):
        director_rating_mean = overall_rating_mean

    if pd.isna(actor_rating_mean):
        actor_rating_mean = overall_rating_mean

    mapped_features = {
        "director_experience": float(
            len(director_data) if not director_data.empty else 1
        ),
        "director_avg_rating": float(director_rating_mean),
        "lead_actor_experience": float(
            len(actor_data) if not actor_data.empty else 1
        ),
        "lead_actor_avg_rating": float(actor_rating_mean),
    }

    for column, value in mapped_features.items():
        if column in input_df.columns:
            input_df.loc[0, column] = value

    genre_data = df[
        df["main_genre"].astype(str).str.strip().str.lower()
        == main_genre
    ]

    if "genre_frequency" in input_df.columns:
        genre_frequency = (
            len(genre_data) / len(df)
            if not genre_data.empty
            else 0.0
        )
        input_df.loc[0, "genre_frequency"] = float(genre_frequency)

    return input_df[model_features].astype(float)


def recommend_movies(
    movie_id: str,
    top_n: int = 5,
) -> pd.DataFrame:
    if movie_id not in movie_indices:
        return pd.DataFrame()

    index_value = movie_indices[movie_id]

    if isinstance(index_value, pd.Series):
        index_value = int(index_value.iloc[0])
    else:
        index_value = int(index_value)

    similarity_scores = np.asarray(
        cosine_sim[index_value]
    ).reshape(-1)

    ordered_indices = similarity_scores.argsort()[::-1]
    ordered_indices = [
        idx
        for idx in ordered_indices
        if idx != index_value
    ][:top_n]

    recommendations = rec_movies.iloc[ordered_indices][
        [
            "movie_id",
            "name",
            "year",
            "genre",
            "director",
            "actor_1",
            "rating",
            "votes",
        ]
    ].copy()

    recommendations["similarity_score"] = (
        similarity_scores[ordered_indices]
    )

    recommendations["similarity_percentage"] = (
        recommendations["similarity_score"] * 100
    ).round(2)

    return recommendations.reset_index(drop=True)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


# ==================================================
# Sidebar navigation
# ==================================================
st.sidebar.title("🎬 CineRate AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict Rating",
        "Recommend Movies",
        "Insights",
        "Model Performance",
    ],
)


# ==================================================
# Home page
# ==================================================
if page == "Home":
    page_header(
        "🎬 CineRate AI",
        "Explainable movie rating prediction and recommendation system",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Movies", f"{len(df):,}")

    with col2:
        st.metric("MAE", "0.431")

    with col3:
        st.metric("RMSE", "0.651")

    with col4:
        st.metric("R² Score", "0.772")

    with st.container(border=True):
        st.subheader("About the project")
        st.write(
            "CineRate AI predicts IMDb-style movie ratings using a "
            "tuned XGBoost regression model and recommends similar "
            "movies using content-based filtering."
        )
        st.write(
            "The project covers data cleaning, exploratory analysis, "
            "feature engineering, model comparison, explainability, "
            "and recommendation systems."
        )

    with st.container(border=True):
        st.subheader("Technology stack")
        st.write(
            "Python · Pandas · NumPy · Scikit-learn · XGBoost · "
            "SHAP · TF-IDF · Cosine Similarity · Streamlit"
        )


# ==================================================
# Predict Rating page
# ==================================================
elif page == "Predict Rating":
    page_header(
        "🎯 Predict Movie Rating",
        "Select a movie and compare its actual and predicted ratings.",
    )

    required_columns = [
        "name",
        "year",
        "duration",
        "votes",
        "main_genre",
        "director",
        "actor_1",
        "rating",
    ]

    prediction_movies = df[required_columns].copy()
    prediction_movies = prediction_movies.dropna(subset=["name"])
    prediction_movies = prediction_movies.sort_values(
        ["name", "year"],
        na_position="last",
    )

    def movie_dropdown_label(index_value: int) -> str:
        movie_name = clean_display_name(
            prediction_movies.loc[index_value, "name"]
        )

        movie_year = prediction_movies.loc[index_value, "year"]

        if pd.notna(movie_year):
            return f"{movie_name} ({int(movie_year)})"

        return movie_name

    selected_index = st.selectbox(
        "🎬 Select a movie",
        options=prediction_movies.index.tolist(),
        format_func=movie_dropdown_label,
    )

    movie = prediction_movies.loc[selected_index]

    year = safe_numeric(
        movie["year"],
        df["year"],
        as_integer=True,
    )

    duration = safe_numeric(
        movie["duration"],
        df["duration"],
        as_integer=True,
    )

    votes = safe_numeric(
        movie["votes"],
        df["votes"],
        as_integer=True,
    )

    genre = (
        str(movie["main_genre"])
        if pd.notna(movie["main_genre"])
        else "Unknown"
    )

    director = (
        str(movie["director"])
        if pd.notna(movie["director"])
        else "Unknown"
    )

    actor_1 = (
        str(movie["actor_1"])
        if pd.notna(movie["actor_1"])
        else "Unknown"
    )

    actual_rating = pd.to_numeric(
        pd.Series([movie["rating"]]),
        errors="coerce",
    ).iloc[0]

    details_col, stats_col = st.columns(2)

    with details_col:
        with st.container(border=True):
            st.subheader("🎥 Movie information")
            st.write(f"**Movie:** {clean_display_name(movie['name'])}")
            st.write(f"**Genre:** {clean_display_name(genre)}")
            st.write(f"**Director:** {clean_display_name(director)}")
            st.write(f"**Lead actor:** {clean_display_name(actor_1)}")

    with stats_col:
        with st.container(border=True):
            st.subheader("📊 Movie statistics")
            st.write(f"**Release year:** {year}")
            st.write(f"**Duration:** {duration} minutes")
            st.write(f"**Votes:** {votes:,}")

            actual_text = (
                f"{actual_rating:.1f} / 10"
                if pd.notna(actual_rating)
                else "N/A"
            )
            st.write(f"**Actual rating:** ⭐ {actual_text}")

    if st.button(
        "🎯 Predict Rating",
        use_container_width=True,
    ):
        input_data = prepare_prediction_input(
            year=year,
            duration=duration,
            votes=votes,
            genre=genre,
            director=director,
            actor_1=actor_1,
        )

        prediction = float(model.predict(input_data)[0])
        prediction = float(np.clip(prediction, 0, 10))

        lower_range = max(0.0, prediction - 0.43)
        upper_range = min(10.0, prediction + 0.43)

        prediction_error = (
            abs(actual_rating - prediction)
            if pd.notna(actual_rating)
            else np.nan
        )

        with st.container(border=True):
            st.subheader(f"🎬 {clean_display_name(movie['name'])}")
            st.metric(
                "Predicted Rating",
                f"{prediction:.2f} / 10",
            )
            st.info(
                f"Expected rating range: "
                f"{lower_range:.2f} – {upper_range:.2f}"
            )
            st.caption(
                "This descriptive range uses the model's mean "
                "absolute error of approximately 0.43 rating points."
            )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Actual Rating",
                (
                    f"{actual_rating:.2f}"
                    if pd.notna(actual_rating)
                    else "N/A"
                ),
            )

        with result_col2:
            st.metric(
                "Predicted Rating",
                f"{prediction:.2f}",
            )

        with result_col3:
            st.metric(
                "Absolute Error",
                (
                    f"{prediction_error:.2f}"
                    if pd.notna(prediction_error)
                    else "N/A"
                ),
            )

        with st.container(border=True):
            st.subheader("💡 Factors used by the model")
            st.write("• Historical performance of the director")
            st.write("• Historical performance of the lead actor")
            st.write("• Vote count and popularity indicators")
            st.write("• Genre, duration, release year, and movie age")

        prediction_report = pd.DataFrame({
            "movie_name": [clean_display_name(movie["name"])],
            "actual_rating": [
                round(actual_rating, 2)
                if pd.notna(actual_rating)
                else np.nan
            ],
            "predicted_rating": [round(prediction, 2)],
            "absolute_error": [
                round(prediction_error, 2)
                if pd.notna(prediction_error)
                else np.nan
            ],
            "expected_lower_rating": [round(lower_range, 2)],
            "expected_upper_rating": [round(upper_range, 2)],
            "genre": [genre],
            "director": [director],
            "lead_actor": [actor_1],
            "release_year": [year],
            "duration_minutes": [duration],
            "votes": [votes],
        })

        safe_filename = "".join(
            character
            if character.isalnum() or character in "-_"
            else "_"
            for character in clean_display_name(movie["name"]).lower()
        ).strip("_")

        st.download_button(
            label="Download Prediction Report",
            data=prediction_report.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=f"{safe_filename}_prediction.csv",
            mime="text/csv",
        )


# ==================================================
# Recommend Movies page
# ==================================================
elif page == "Recommend Movies":
    page_header(
        "🎬 Recommend Similar Movies",
        "Select a movie to discover similar titles.",
    )

    movie_list = sorted(
        rec_movies["movie_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_movie = st.selectbox(
        "Choose a movie",
        movie_list,
        format_func=clean_display_name,
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
    )

    if st.button(
        "Get Recommendations",
        use_container_width=True,
    ):
        recommendations = recommend_movies(
            selected_movie,
            top_n,
        )

        if recommendations.empty:
            st.warning("No recommendations were found.")

        else:
            st.subheader(
                f"Recommendations for {clean_display_name(selected_movie)}"
            )

            for _, row in recommendations.iterrows():
                year_text = (
                    str(int(row["year"]))
                    if pd.notna(row["year"])
                    else "N/A"
                )

                votes_text = (
                    f"{int(row['votes']):,}"
                    if pd.notna(row["votes"])
                    else "N/A"
                )

                rating_text = (
                    f"{float(row['rating']):.1f}"
                    if pd.notna(row["rating"])
                    else "N/A"
                )

                similarity = float(
                    np.clip(
                        row["similarity_score"],
                        0,
                        1,
                    )
                )

                with st.container(border=True):
                    st.subheader(
                        f"{clean_display_name(row['name'])} ({year_text})"
                    )

                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        st.write(
                            f"**Genre:** "
                            f"{clean_display_name(row['genre'])}"
                        )
                        st.write(
                            f"**Director:** "
                            f"{clean_display_name(row['director'])}"
                        )
                        st.write(
                            f"**Lead actor:** "
                            f"{clean_display_name(row['actor_1'])}"
                        )

                    with info_col2:
                        st.metric("Rating", f"⭐ {rating_text}")
                        st.write(f"**Votes:** {votes_text}")
                        st.write(
                            f"**Similarity:** "
                            f"{float(row['similarity_percentage']):.2f}%"
                        )

                    st.progress(similarity)


# ==================================================
# Insights page
# ==================================================
elif page == "Insights":
    page_header(
        "📊 Dataset Insights",
        "Explore important patterns in the movie dataset.",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.subheader("Rating Distribution")

            figure, axis = plt.subplots(figsize=(6, 4))
            axis.hist(
                pd.to_numeric(
                    df["rating"],
                    errors="coerce",
                ).dropna(),
                bins=30,
            )
            axis.set_xlabel("Rating")
            axis.set_ylabel("Number of Movies")
            axis.set_title("Distribution of Movie Ratings")
            st.pyplot(figure)
            plt.close(figure)

    with chart_col2:
        with st.container(border=True):
            st.subheader("Votes vs Rating")

            chart_data = df[["votes", "rating"]].copy()
            chart_data["votes"] = pd.to_numeric(
                chart_data["votes"],
                errors="coerce",
            )
            chart_data["rating"] = pd.to_numeric(
                chart_data["rating"],
                errors="coerce",
            )
            chart_data = chart_data.dropna()

            figure, axis = plt.subplots(figsize=(6, 4))
            axis.scatter(
                chart_data["votes"],
                chart_data["rating"],
                alpha=0.4,
            )
            axis.set_xlabel("Votes")
            axis.set_ylabel("Rating")
            axis.set_title("Votes Compared with Rating")
            st.pyplot(figure)
            plt.close(figure)

    genre_col1, genre_col2 = st.columns(2)

    with genre_col1:
        with st.container(border=True):
            st.subheader("Top Genres by Movie Count")
            genre_counts = (
                df["main_genre"]
                .dropna()
                .value_counts()
                .head(10)
            )
            st.bar_chart(genre_counts)

    with genre_col2:
        with st.container(border=True):
            st.subheader("Average Rating by Genre")

            genre_rating_data = df[
                ["main_genre", "rating"]
            ].copy()

            genre_rating_data["rating"] = pd.to_numeric(
                genre_rating_data["rating"],
                errors="coerce",
            )

            genre_rating = (
                genre_rating_data
                .dropna()
                .groupby("main_genre")["rating"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )

            st.bar_chart(genre_rating)


# ==================================================
# Model Performance page
# ==================================================
elif page == "Model Performance":
    page_header(
        "📈 Model Performance",
        "Evaluation results for the tuned XGBoost model.",
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("MAE", "0.431")

    with metric_col2:
        st.metric("RMSE", "0.651")

    with metric_col3:
        st.metric("R² Score", "0.772")

    with metric_col4:
        st.metric("CV RMSE", "0.650 ± 0.012")

    with st.container(border=True):
        st.subheader("Final model")
        st.write(
            "The tuned **XGBoost Regressor** performed best "
            "among the models evaluated during training."
        )
        st.write(
            "It explains approximately **77.2%** of the "
            "variation in movie ratings."
        )
        st.write(
            "Five-fold cross-validation produced a mean RMSE "
            "of 0.650 with a standard deviation of 0.012."
        )

    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        feature_importance = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

        with st.container(border=True):
            st.subheader("Top Feature Importances")

            st.dataframe(
                feature_importance.head(20),
                use_container_width=True,
                hide_index=True,
            )

            if {
                "Feature",
                "Importance",
            }.issubset(feature_importance.columns):
                top_features = (
                    feature_importance
                    .sort_values(
                        "Importance",
                        ascending=False,
                    )
                    .head(15)
                    .sort_values(
                        "Importance",
                        ascending=True,
                    )
                )

                figure, axis = plt.subplots(
                    figsize=(8, 6)
                )

                axis.barh(
                    top_features["Feature"],
                    top_features["Importance"],
                )

                axis.set_xlabel("Importance")
                axis.set_title("Top 15 Important Features")

                st.pyplot(figure)
                plt.close(figure)


# ==================================================
# Footer
# ==================================================
st.divider()
st.caption("Built by Nikhita Darshanala · CineRate AI")
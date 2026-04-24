# analyse_dais_cluster.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILE = "DAIS Research Group Survey_ Data & NLP Cluster Identity Mapping.xlsx"


def load_data():
    df = pd.read_excel(FILE)
    df = df.drop(columns=["Start time", "Completion time", "Email"], errors="ignore")
    return df


def rename_columns(df):
    df.columns = [
        "id",
        "name",
        "research_description",
        "publications",
        "cluster_connection",
        "future_directions",
        "human_ai_scope",
        "missing_topics",
        "expected_benefits",
        "meeting_frequency",
        "meeting_activities",
        "first_meeting_activity",
    ]
    return df


def basic_summary(df):
    print("\n--- MEMBERS ---")
    print(df["name"].tolist())

    print("\n--- MEETING FREQUENCY PREFERENCES ---")
    print(df["meeting_frequency"].value_counts())

    print("\n--- EXPECTED BENEFITS ---")
    print(df["expected_benefits"].value_counts())


def extract_keywords(df, column, top_n=10):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200
    )

    tfidf = vectorizer.fit_transform(df[column].fillna(""))
    words = vectorizer.get_feature_names_out()

    scores = np.asarray(tfidf.mean(axis=0)).flatten()

    ranked = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)

    print(f"\n--- TOP THEMES FROM: {column} ---")
    for w, s in ranked[:top_n]:
        print(w, round(s, 3))


def similarity_matrix(df):

    vectorizer = TfidfVectorizer(stop_words="english")

    text = (
        df["research_description"].fillna("")
        + " "
        + df["cluster_connection"].fillna("")
    )

    tfidf = vectorizer.fit_transform(text)

    similarity = cosine_similarity(tfidf)

    similarity_df = pd.DataFrame(
        similarity,
        index=df["name"],
        columns=df["name"]
    )

    print("\n--- RESEARCH SIMILARITY MATRIX ---")
    print(similarity_df.round(2))

    similarity_df.to_csv("research_similarity_matrix.csv")


def meeting_activity_themes(df):

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf = vectorizer.fit_transform(
        df["meeting_activities"].fillna("")
    )

    words = vectorizer.get_feature_names_out()

    scores = np.asarray(tfidf.mean(axis=0)).flatten()

    ranked = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)

    print("\n--- MEETING ACTIVITY PRIORITIES ---")

    for w, s in ranked[:10]:
        print(w, round(s, 3))


def future_direction_themes(df):

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf = vectorizer.fit_transform(
        df["future_directions"].fillna("")
    )

    words = vectorizer.get_feature_names_out()

    scores = np.asarray(tfidf.mean(axis=0)).flatten()

    ranked = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)

    print("\n--- FUTURE RESEARCH PRIORITIES ---")

    for w, s in ranked[:10]:
        print(w, round(s, 3))


def main():

    df = load_data()
    df = rename_columns(df)

    basic_summary(df)

    extract_keywords(df, "research_description")
    extract_keywords(df, "cluster_connection")

    future_direction_themes(df)

    meeting_activity_themes(df)

    similarity_matrix(df)


if __name__ == "__main__":
    main()
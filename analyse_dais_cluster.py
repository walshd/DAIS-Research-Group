# analyse_dais_cluster.py

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILE = "DAIS Research Group Survey_ Data & NLP Cluster Identity Mapping.xlsx"
REPORT_FILE = "survey_question_summaries.md"
THEME_EXTRACTION_MODE = "dynamic"

QUESTION_COLUMNS = [
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

QUESTION_LABELS = {
    "research_description": "What best describes your current research area and methods?",
    "publications": "What publication venues, outputs, or publication interests are most relevant to your work?",
    "cluster_connection": "How does your work connect with the DAIS Data & NLP cluster?",
    "future_directions": "What future research directions would you like the cluster to focus on?",
    "human_ai_scope": "How do you see Human-AI issues fitting into your work?",
    "missing_topics": "What topics are currently missing from the cluster and should be added?",
    "expected_benefits": "What benefits do you expect from participating in the cluster?",
    "meeting_frequency": "How often should the group meet?",
    "meeting_activities": "What meeting activities should be prioritised?",
    "first_meeting_activity": "What should the first meeting focus on?",
}

CATEGORICAL_COLUMNS = {"meeting_frequency"}
NOISY_TERMS = {
    "yes",
    "using",
    "use",
    "work",
    "works",
    "like",
    "would",
    "also",
    "particularly",
}

IDENTITY_TEXT_COLUMNS = [
    "research_description",
    "publications",
    "cluster_connection",
    "future_directions",
    "human_ai_scope",
    "missing_topics",
]

THEME_DEFINITIONS = {
    "core_methods": {
        "label": "Core methodological spine: data-driven intelligent systems",
        "min_keyword_hits": 2,
        "keywords": [
            "machine learning",
            "ml",
            "nlp",
            "natural language",
            "multimodal",
            "signal processing",
            "predictive",
            "modelling",
            "modeling",
            "classification",
            "intelligent systems",
            "data-driven",
            "data driven",
        ],
        "narrative": (
            "The strongest overlap is methodological: members consistently describe "
            "data-driven intelligent analysis workflows rather than a loose AI interest group."
        ),
    },
    "multimodal_capability": {
        "label": "Strong multimodal capability",
        "min_keyword_hits": 2,
        "keywords": [
            "text",
            "tabular",
            "image",
            "images",
            "eeg",
            "neuro",
            "signal",
            "social media",
            "cultural heritage",
            "multimodal",
            "multi-modal",
            "datasets",
        ],
        "narrative": (
            "Responses indicate coverage across multiple data modalities, suggesting "
            "cross-modality modelling capacity rather than single-modality analysis."
        ),
    },
    "human_ai_layer": {
        "label": "Human-centred interpretation and interaction",
        "min_keyword_hits": 2,
        "keywords": [
            "human",
            "human-ai",
            "human ai",
            "interaction",
            "interpret",
            "interpretation",
            "interface",
            "real-world",
            "real world",
            "ethics",
            "evaluation",
            "meaning",
        ],
        "narrative": (
            "A recurring signal is that AI outputs need human interpretation, interaction, "
            "and evaluation in applied settings."
        ),
    },
    "domain_transferability": {
        "label": "Domain diversity with structural coherence",
        "min_keyword_hits": 1,
        "keywords": [
            "health",
            "healthcare",
            "education",
            "policy",
            "social media",
            "cultural heritage",
            "behaviour",
            "behavior",
            "cognitive",
            "society",
            "societal",
        ],
        "narrative": (
            "Application domains differ, but they are structurally similar as complex "
            "human-centred data environments, supporting transfer of methods across contexts."
        ),
    },
    "language_anchor": {
        "label": "Language as a unifying anchor modality",
        "min_keyword_hits": 2,
        "keywords": [
            "nlp",
            "language",
            "linguistic",
            "document",
            "summarisation",
            "summarization",
            "classification",
            "fake news",
            "metadata",
            "llm",
            "text",
        ],
        "narrative": (
            "Language appears repeatedly as an input, analytic, or interpretive layer, "
            "providing a coherent centre of gravity for cluster identity."
        ),
    },
}

PILLAR_MAP = [
    ("Pillar 1", "Multimodal intelligent data analysis", ["core_methods", "multimodal_capability"]),
    ("Pillar 2", "Language-centred AI and NLP methods", ["language_anchor", "core_methods"]),
    ("Pillar 3", "Human interpretation, interaction, and evaluation of AI systems", ["human_ai_layer"]),
    ("Pillar 4", "Applied intelligent systems in complex human domains", ["domain_transferability", "human_ai_layer"]),
]

PILLAR_KEYWORDS = {
    "Pillar 1": ["multimodal", "signal", "image", "tabular", "data", "modelling", "modeling", "analysis"],
    "Pillar 2": ["nlp", "language", "linguistic", "document", "llm", "text", "classification"],
    "Pillar 3": ["human", "interaction", "interpretation", "evaluation", "ethics", "interface"],
    "Pillar 4": ["health", "education", "policy", "social", "cultural", "applied", "real-world"],
}

PILLAR_TITLES = {
    "Pillar 1": "Multimodal intelligent data analysis",
    "Pillar 2": "Language-centred AI and NLP methods",
    "Pillar 3": "Human interpretation, interaction, and evaluation of AI systems",
    "Pillar 4": "Applied intelligent systems in complex human domains",
}


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


def _clean_text_series(series):
    text = series.dropna().astype(str).str.strip()
    text = text[text != ""]
    return text


def _format_top_items(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _normalise_text(text):
    clean = re.sub(r"\s+", " ", str(text).strip().lower())
    return clean


def _theme_hits_for_text(text, keywords):
    hits = []
    for keyword in keywords:
        if " " in keyword or "-" in keyword:
            pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
        else:
            pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, text):
            hits.append(keyword)
    return hits


def build_identity_synthesis(df):
    if THEME_EXTRACTION_MODE == "dynamic":
        return build_identity_synthesis_dynamic(df)
    return build_identity_synthesis_predefined(df)


def _build_pillars_from_dynamic_signals(signal_blocks):
    pillars = []
    for pillar_name, keywords in PILLAR_KEYWORDS.items():
        best_score = 0.0
        for signal in signal_blocks:
            terms = signal["evidence_terms"]
            if not terms:
                continue
            matches = sum(1 for kw in keywords if any(kw in term for term in terms))
            match_ratio = matches / len(keywords)
            weighted = match_ratio * signal["coverage"]
            best_score = max(best_score, weighted)
        pillars.append((pillar_name, PILLAR_TITLES[pillar_name], best_score))
    pillars.sort(key=lambda item: item[2], reverse=True)
    return pillars


def _generate_dynamic_signal_narrative(title, top_terms):
    terms_text = _format_top_items(top_terms[:4])
    return (
        f"This emergent theme is characterised by repeated references to {terms_text}, "
        "indicating a coherent shared direction across contributors."
    )


def _build_name_stopwords(df):
    if "name" not in df.columns:
        return set()
    names = df["name"].dropna().astype(str).tolist()
    tokens = set()
    for full_name in names:
        parts = re.findall(r"[a-zA-Z]+", full_name.lower())
        for part in parts:
            if len(part) > 2:
                tokens.add(part)
    return tokens


def build_identity_synthesis_dynamic(df):
    combined = (
        df[IDENTITY_TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(_normalise_text)
    )
    combined = combined[combined != ""]

    total_people = len(combined)
    if total_people == 0:
        return {"signals": [], "pillars": [], "total_people": 0, "mode": "dynamic"}

    n_topics = min(4, max(2, total_people // 2))
    dynamic_stop_words = set(ENGLISH_STOP_WORDS) | _build_name_stopwords(df)
    min_df = 2 if total_people >= 5 else 1
    vectorizer = TfidfVectorizer(
        stop_words=list(dynamic_stop_words),
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9,
    )
    try:
        tfidf = vectorizer.fit_transform(combined)
    except ValueError:
        return {"signals": [], "pillars": [], "total_people": total_people, "mode": "dynamic"}

    if tfidf.shape[1] < 2:
        return {"signals": [], "pillars": [], "total_people": total_people, "mode": "dynamic"}

    n_topics = min(n_topics, max(1, tfidf.shape[1] - 1))
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=500)
    topic_weights = nmf.fit_transform(tfidf)
    topic_terms = nmf.components_
    features = np.array(vectorizer.get_feature_names_out())

    row_sums = topic_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    topic_shares = topic_weights / row_sums

    signal_blocks = []
    for topic_idx in range(n_topics):
        top_term_idx = np.argsort(topic_terms[topic_idx])[::-1][:8]
        top_terms = [features[i] for i in top_term_idx if topic_terms[topic_idx][i] > 0]

        if not top_terms:
            continue

        coverage_mask = topic_shares[:, topic_idx] >= 0.20
        covered = int(np.sum(coverage_mask))
        coverage = covered / total_people if total_people else 0

        evidence_terms = top_terms[:5]
        evidence_text = _format_top_items([f"{term}" for term in evidence_terms])
        title = f"Emergent theme: {_format_top_items(evidence_terms[:3])}"

        signal_blocks.append(
            {
                "title": title,
                "coverage": coverage,
                "covered": covered,
                "total": total_people,
                "narrative": _generate_dynamic_signal_narrative(title, evidence_terms),
                "evidence_text": evidence_text,
                "evidence_terms": evidence_terms,
            }
        )

    signal_blocks = [s for s in signal_blocks if s["covered"] > 0]
    signal_blocks.sort(key=lambda item: item["coverage"], reverse=True)

    pillars = _build_pillars_from_dynamic_signals(signal_blocks)
    return {
        "signals": signal_blocks,
        "pillars": pillars,
        "total_people": total_people,
        "mode": "dynamic",
    }


def build_identity_synthesis_predefined(df):
    combined = (
        df[IDENTITY_TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(_normalise_text)
    )

    total_people = len(combined)
    signal_blocks = []
    theme_scores = {}

    for theme_id, theme in THEME_DEFINITIONS.items():
        person_hits = []
        matched_terms = []
        min_hits = int(theme.get("min_keyword_hits", 1))

        for text in combined:
            hits = _theme_hits_for_text(text, theme["keywords"])
            if len(set(hits)) >= min_hits:
                person_hits.append(1)
                matched_terms.extend(hits)
            else:
                person_hits.append(0)

        covered = int(np.sum(person_hits))
        coverage = covered / total_people if total_people else 0

        term_counts = pd.Series(matched_terms).value_counts() if matched_terms else pd.Series(dtype="int64")
        evidence_terms = term_counts.head(5).index.tolist()

        theme_scores[theme_id] = coverage

        evidence_text = (
            _format_top_items([f"{term} ({term_counts[term]})" for term in evidence_terms])
            if evidence_terms
            else "no repeated evidence terms"
        )

        signal_blocks.append(
            {
                "title": theme["label"],
                "coverage": coverage,
                "covered": covered,
                "total": total_people,
                "narrative": theme["narrative"],
                "evidence_text": evidence_text,
            }
        )

    signal_blocks.sort(key=lambda item: item["coverage"], reverse=True)

    pillars = []
    for pillar_name, pillar_label, linked_themes in PILLAR_MAP:
        strength = float(np.mean([theme_scores[t] for t in linked_themes])) if linked_themes else 0.0
        pillars.append((pillar_name, pillar_label, strength))
    pillars.sort(key=lambda item: item[2], reverse=True)

    return {
        "signals": signal_blocks,
        "pillars": pillars,
        "total_people": total_people,
        "mode": "predefined",
    }


def summarize_categorical_question(df, column, top_n=3):
    values = _clean_text_series(df[column])
    total = len(values)

    if total == 0:
        return "No responses were provided for this question."

    counts = values.value_counts()
    top = counts.head(top_n)

    highlights = [
        f"{label} ({count}/{total}, {count / total:.0%})"
        for label, count in top.items()
    ]

    lead = top.index[0]
    return (
        f"Across {total} responses, the most common choice was {lead}. "
        f"The strongest preferences were {_format_top_items(highlights)}, "
        "so these should be treated as the highest-priority options in planning."
    )


def summarize_text_question(df, column, top_n=5):
    values = _clean_text_series(df[column])
    total = len(values)

    if total == 0:
        return "No responses were provided for this question."

    min_df = 2 if total >= 5 else 1
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df)
    matrix = vectorizer.fit_transform(values)

    terms = vectorizer.get_feature_names_out()
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    term_freq = np.asarray(matrix.sum(axis=0)).ravel()

    keep_mask = np.array([term not in NOISY_TERMS for term in terms])
    terms = terms[keep_mask]
    doc_freq = doc_freq[keep_mask]
    term_freq = term_freq[keep_mask]

    if len(terms) == 0:
        return (
            f"From {total} responses, there was no single dominant repeated phrase. "
            "Answers were diverse, so this question should be interpreted as containing "
            "multiple priorities rather than one clear top theme."
        )

    ranked_idx = np.lexsort((-term_freq, -doc_freq))
    top_idx = ranked_idx[:top_n]

    if doc_freq[top_idx[0]] < 2:
        return (
            f"From {total} responses, no specific theme was repeated often enough to "
            "dominate. This indicates varied perspectives, so the response set should "
            "be treated as a broad agenda with multiple equally important ideas."
        )

    top_themes = [
        f"{terms[i]} ({doc_freq[i]}/{total} responses, {doc_freq[i] / total:.0%})"
        for i in top_idx
    ]

    return (
        f"From {total} responses, the most repeated themes were "
        f"{_format_top_items(top_themes)}. "
        "These recurring themes appear most often across submissions and should be "
        "treated as the most important priorities for this question."
    )


def _top_terms_from_columns(df, columns, top_n=4):
    text_chunks = []
    for column in columns:
        if column in df.columns:
            text_chunks.append(df[column].fillna("").astype(str))

    if not text_chunks:
        return []

    combined = pd.concat(text_chunks, axis=1).agg(" ".join, axis=1)
    values = _clean_text_series(combined)
    total = len(values)
    if total == 0:
        return []

    min_df = 2 if total >= 5 else 1
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df)
    matrix = vectorizer.fit_transform(values)

    terms = vectorizer.get_feature_names_out()
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    term_freq = np.asarray(matrix.sum(axis=0)).ravel()

    keep_mask = np.array([term not in NOISY_TERMS for term in terms])
    terms = terms[keep_mask]
    doc_freq = doc_freq[keep_mask]
    term_freq = term_freq[keep_mask]
    if len(terms) == 0:
        return []

    ranked_idx = np.lexsort((-term_freq, -doc_freq))
    top_idx = ranked_idx[:top_n]
    return [(terms[i], int(doc_freq[i]), total) for i in top_idx]


def _format_term_with_share(items):
    if not items:
        return ""
    formatted = [f"{term} ({count}/{total}, {count / total:.0%})" for term, count, total in items]
    return _format_top_items(formatted)


def generate_group_identity_paragraph(df, identity_synthesis):
    total_members = len(df)

    methods = _top_terms_from_columns(df, ["research_description", "cluster_connection", "publications"], top_n=4)
    future = _top_terms_from_columns(df, ["future_directions", "human_ai_scope", "missing_topics"], top_n=4)
    collaboration = _top_terms_from_columns(df, ["expected_benefits", "meeting_activities", "first_meeting_activity"], top_n=4)

    frequency_values = _clean_text_series(df["meeting_frequency"]) if "meeting_frequency" in df.columns else pd.Series(dtype=str)
    if len(frequency_values) > 0:
        top_freq = frequency_values.value_counts().head(2)
        frequency_text = _format_top_items([
            f"{label} ({count}/{len(frequency_values)}, {count / len(frequency_values):.0%})"
            for label, count in top_freq.items()
        ])
    else:
        frequency_text = "no clear preferred cadence"

    top_signals = identity_synthesis.get("signals", [])[:2]
    if top_signals:
        signal_text = _format_top_items([
            f"{signal['title'].replace('Emergent theme: ', '').lower()} ({signal['covered']}/{signal['total']}, {signal['coverage']:.0%})"
            for signal in top_signals
        ])
    else:
        signal_text = "no dominant cross-question identity signals"

    methods_text = _format_term_with_share(methods) if methods else "diverse but related methodological approaches"
    future_text = _format_term_with_share(future) if future else "a broad agenda for future research"
    collaboration_text = _format_term_with_share(collaboration) if collaboration else "shared interest in collaboration and planning"

    return (
        f"Across {total_members} contributors, the group presents a coherent identity with strongest shared signals in {signal_text}. "
        f"Methodologically, responses repeatedly emphasise {methods_text}, indicating a data- and language-oriented research core. "
        f"Future priorities are concentrated around {future_text}, while members describe collaboration needs around {collaboration_text}. "
        f"Operationally, the preferred meeting cadence clusters around {frequency_text}, supporting a regular and action-focused community model."
    )


def generate_question_paragraphs(df):
    paragraphs = {}
    for column in QUESTION_COLUMNS:
        if column in CATEGORICAL_COLUMNS:
            paragraph = summarize_categorical_question(df, column)
        else:
            paragraph = summarize_text_question(df, column)
        paragraphs[column] = paragraph
    return paragraphs


def write_summary_report(paragraphs, identity_synthesis, group_identity_paragraph):
    mode_text = "dynamic topic modelling" if identity_synthesis.get("mode") == "dynamic" else "predefined thematic dictionaries"
    lines = [
        "# DAIS Survey Summary (Reproducible)",
        "",
        "This report is automatically generated from all submitted responses.",
        "Items that appear most often are highlighted as the highest priorities.",
        f"Identity synthesis mode: {mode_text}.",
        "",
        "## Cluster Identity Synthesis",
        "",
        (
            "Across responses, the strongest overlap appears at the methodological and "
            "human-AI layer. The identity signals below are ranked by respondent coverage "
            f"out of {identity_synthesis['total_people']} contributors."
        ),
        "",
        "## Group Identity Paragraph",
        "",
        group_identity_paragraph,
        "",
    ]

    for idx, signal in enumerate(identity_synthesis["signals"], start=1):
        lines.append(f"### {idx}. {signal['title']}")
        lines.append("")
        lines.append(
            f"Coverage: {signal['covered']}/{signal['total']} responses ({signal['coverage']:.0%}). "
            f"{signal['narrative']}"
        )
        lines.append("")
        lines.append(f"Evidence terms: {signal['evidence_text']}.")
        lines.append("")

    lines.append("## Candidate Research Pillars")
    lines.append("")
    for pillar_name, pillar_label, strength in identity_synthesis["pillars"]:
        lines.append(f"- {pillar_name}: {pillar_label} (support score: {strength:.0%})")
    lines.append("")

    lines.append("## Per-question Summaries")
    lines.append("")

    for column in QUESTION_COLUMNS:
        question = QUESTION_LABELS.get(column, column)
        lines.append(f"## {question}")
        lines.append("")
        lines.append(paragraphs[column])
        lines.append("")

    with open(REPORT_FILE, "w", encoding="utf-8") as output:
        output.write("\n".join(lines))


def similarity_matrix(df):
    vectorizer = CountVectorizer(stop_words="english")

    text = df["research_description"].fillna("") + " " + df["cluster_connection"].fillna("")
    tf = vectorizer.fit_transform(text)

    similarity = cosine_similarity(tf)
    similarity_df = pd.DataFrame(similarity, index=df["name"], columns=df["name"])
    similarity_df.to_csv("research_similarity_matrix.csv")
    return similarity_df


def main():
    df = load_data()
    df = rename_columns(df)

    paragraphs = generate_question_paragraphs(df)
    identity_synthesis = build_identity_synthesis(df)
    group_identity_paragraph = generate_group_identity_paragraph(df, identity_synthesis)
    write_summary_report(paragraphs, identity_synthesis, group_identity_paragraph)

    similarity_df = similarity_matrix(df)

    print("\n--- SURVEY QUESTION SUMMARIES ---")
    for column in QUESTION_COLUMNS:
        question = QUESTION_LABELS.get(column, column)
        print(f"\n{question}")
        print(paragraphs[column])

    print("\n--- RESEARCH SIMILARITY MATRIX ---")
    print(similarity_df.round(2))
    print("\n--- GROUP IDENTITY PARAGRAPH ---")
    print(group_identity_paragraph)
    print("\n--- CLUSTER IDENTITY SIGNALS ---")
    for idx, signal in enumerate(identity_synthesis["signals"], start=1):
        print(
            f"{idx}. {signal['title']} - {signal['covered']}/{signal['total']} "
            f"({signal['coverage']:.0%})"
        )
    print("\n--- CANDIDATE RESEARCH PILLARS ---")
    for pillar_name, pillar_label, strength in identity_synthesis["pillars"]:
        print(f"{pillar_name}: {pillar_label} ({strength:.0%})")
    print(f"\nSaved summary report to: {REPORT_FILE}")
    print("Saved similarity matrix to: research_similarity_matrix.csv")


if __name__ == "__main__":
    main()
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline


def build_standard_pipeline() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                dtype="float32",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def build_streaming_pipeline() -> Pipeline:
    return Pipeline([
        (
            "vectorizer",
            HashingVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                n_features=2**18,
                alternate_sign=False,
                norm="l2",
            ),
        ),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss",
                random_state=42,
            ),
        ),
    ])

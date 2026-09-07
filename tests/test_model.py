from src.preprocessing.pipeline import (
    build_standard_pipeline,
    build_streaming_pipeline,
)


def test_standard_pipeline_steps() -> None:
    pipeline = build_standard_pipeline()

    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_streaming_pipeline_steps() -> None:
    pipeline = build_streaming_pipeline()

    assert "vectorizer" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
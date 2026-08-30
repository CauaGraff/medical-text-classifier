from src.preprocessing.pipeline import build_pipeline

def test_pipeline_steps() -> None:
    pipeline = build_pipeline()
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

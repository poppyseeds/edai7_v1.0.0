from app.text import TemplateTextGenerator, TextGenerationConfig, evaluate_consistency


def test_local_template_and_consistency_are_auditable():
    records = [{"status": "normal", "temperature": 21}, {"status": "alert", "temperature": 88}]
    texts = TemplateTextGenerator(TextGenerationConfig(fields=("status", "temperature"))).generate(records)
    report = evaluate_consistency(records, texts, {"status": {"normal": "normal", "alert": "alert"}})
    assert report["consistency_rate"] == 1.0
    assert report["method"].startswith("case-insensitive")


def test_consistency_reports_missing_expected_fact():
    report = evaluate_consistency([{"status": "alert"}], ["status is normal"], {"status": {"alert": "alert"}})
    assert report["consistency_rate"] == 0
    assert report["records"][0]["violations"][0]["field"] == "status"

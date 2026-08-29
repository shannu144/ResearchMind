import os
import tempfile
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.services.data_science.corpus_analyzer import CorpusAnalyzer
from app.services.data_science.eda_engine import EDAEngine


def test_corpus_analyzer():
    doc1 = Document(
        id=1,
        filename="paper1.txt",
        file_type="txt",
        file_path="/tmp/paper1.txt",
        file_size=100,
        page_count=1,
        word_count=10,
    )
    page1 = DocumentPage(
        id=1,
        document_id=1,
        page_number=1,
        raw_text="Machine learning models process dataset features and generate predictions.",
        cleaned_text="machine learning models process dataset features generate predictions",
        word_count=8,
    )

    analyzer = CorpusAnalyzer()
    summary = analyzer.analyze_corpus([doc1], [page1])

    assert summary.total_documents == 1
    assert summary.total_pages == 1
    assert summary.vocabulary_size > 0
    assert summary.type_token_ratio > 0.0

    ngrams = analyzer.generate_ngram_analysis([page1])
    assert len(ngrams.unigrams) > 0
    assert len(ngrams.bigrams) > 0
    assert any(b.term == "machine learning" for b in ngrams.bigrams)


def test_eda_engine():
    csv_data = "x,y,category\n1.0,2.0,A\n2.0,4.0,A\n3.0,6.0,B\n4.0,8.0,B\n5.0,10.0,C\n"
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(csv_data)
        tmp_path = tmp.name

    try:
        engine = EDAEngine()
        res = engine.analyze_dataset(tmp_path, document_id=1, filename="dataset.csv")

        assert res.row_count == 5
        assert res.column_count == 3
        assert "x" in res.numerical_columns
        assert "y" in res.numerical_columns
        assert "category" in res.categorical_columns

        # Verify Pearson correlation matrix (x and y are perfectly linearly correlated: r = 1.0)
        assert res.correlation_matrix is not None
        assert len(res.correlation_matrix.columns) == 2
        assert res.correlation_matrix.matrix[0][1] == 1.0

        # Verify categorical distribution
        assert "category" in res.categorical_distributions
        cat_items = res.categorical_distributions["category"]
        assert len(cat_items) == 3
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_data_science_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        summary_resp = await ac.get("/api/v1/data-science/corpus-summary")
        assert summary_resp.status_code == 200
        summary_data = summary_resp.json()
        assert "vocabulary_size" in summary_data

        ngram_resp = await ac.get("/api/v1/data-science/ngrams")
        assert ngram_resp.status_code == 200
        ngram_data = ngram_resp.json()
        assert "unigrams" in ngram_data
        assert "bigrams" in ngram_data

import os
import tempfile
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.document_ingestion.parsers import (
    TXTParser,
    CSVParser,
    DocumentParserFactory,
)
from app.services.data_processing.text_processor import TextProcessor
from app.services.data_processing.csv_processor import CSVProcessor


def test_txt_parser():
    content = "ResearchMind is an AI platform.\nIt processes research papers and datasets efficiently."
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parser = TXTParser()
        parsed = parser.parse(tmp_path, "sample_paper.txt")
        assert parsed.filename == "sample_paper.txt"
        assert parsed.metadata.page_count >= 1
        assert parsed.metadata.total_word_count == 12
        assert len(parsed.pages) >= 1
        assert "ResearchMind" in parsed.pages[0].raw_text
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_csv_parser_and_processor():
    csv_content = "id,age,score\n1,25,85.5\n2,30,90.0\n3,35,95.5\n4,100,20.0\n"
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(csv_content)
        tmp_path = tmp.name

    try:
        parser = CSVParser()
        parsed = parser.parse(tmp_path, "test_data.csv")
        assert parsed.metadata.file_type == "csv"
        assert parsed.metadata.extra_metadata["row_count"] == 4

        processor = CSVProcessor()
        res = processor.analyze_csv(tmp_path)
        assert res.row_count == 4
        assert res.column_count == 3
        assert "age" in res.numerical_columns
        assert "score" in res.numerical_columns
        assert res.missing_values["age"] == 0
        assert res.summary_statistics["age"]["mean"] == 47.5
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_text_processor():
    processor = TextProcessor()
    text = "The ResearchMind platform uses Natural Language Processing and Deep Learning! Is it effective? Yes."
    
    sentences = processor.segment_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The ResearchMind platform uses Natural Language Processing and Deep Learning!"

    res = processor.process_text(text)
    assert res.word_count > 0
    # Stopwords like 'the', 'and', 'is', 'it' should be filtered
    assert "the" not in res.tokens
    assert "and" not in res.tokens
    assert "researchmind" in res.tokens
    assert "learning" in res.tokens


@pytest.mark.asyncio
async def test_document_upload_and_list_api():
    content = b"Title: Machine Learning Advances\nAuthor: Dr. Smith\nThis paper discusses Transformers and RAG systems."
    files = {"file": ("paper.txt", content, "text/plain")}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        upload_resp = await ac.post("/api/v1/documents/upload", files=files)
        assert upload_resp.status_code == 201
        doc_data = upload_resp.json()
        assert doc_data["filename"] == "paper.txt"
        doc_id = doc_data["id"]

        list_resp = await ac.get("/api/v1/documents")
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert any(d["id"] == doc_id for d in docs)

        detail_resp = await ac.get(f"/api/v1/documents/{doc_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert len(detail["pages"]) >= 1
        assert "Transformers" in detail["pages"][0]["raw_text"]

        process_resp = await ac.post(f"/api/v1/documents/{doc_id}/process")
        assert process_resp.status_code == 200
        proc_data = process_resp.json()
        assert proc_data["status"] == "processed"

        del_resp = await ac.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 204

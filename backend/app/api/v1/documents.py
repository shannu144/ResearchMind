import os
import shutil
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.config import settings
from app.core.logging import logger
from app.database.session import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.schemas.document_schemas import (
    DocumentRead,
    DocumentDetailRead,
    DocumentPageRead,
    TextPreprocessingConfig,
    TextPreprocessingResult,
    CSVPreprocessingResult,
)
from app.services.document_ingestion.parsers import DocumentParserFactory
from app.services.data_processing.text_processor import TextProcessor
from app.services.data_processing.csv_processor import CSVProcessor

router = APIRouter(prefix="/documents", tags=["Documents"])
text_processor = TextProcessor()
csv_processor = CSVProcessor()


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document (PDF, DOCX, TXT, CSV), parse text & page numbers, extract metadata, and store in DB.
    """
    filename = file.filename or "uploaded_file.txt"
    ext = os.path.splitext(filename)[1].lower().replace(".", "")
    allowed_extensions = ["pdf", "docx", "doc", "txt", "csv", "md"]

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Supported formats: {', '.join(allowed_extensions)}",
        )

    # Save raw file to RAW_DATA_DIR
    target_dir = settings.RAW_DATA_DIR
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)

    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        logger.error(f"Failed to save file {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    file_size = os.path.getsize(file_path)

    # Parse document using parser strategy
    try:
        parser = DocumentParserFactory.get_parser(filename)
        parsed_doc = parser.parse(file_path, filename)
    except Exception as e:
        logger.error(f"Document parsing failed for {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error parsing file content: {str(e)}",
        )

    # Create Document DB Model
    doc_model = Document(
        filename=filename,
        title=parsed_doc.metadata.title,
        author=parsed_doc.metadata.author,
        file_type=ext,
        file_path=file_path,
        file_size=file_size,
        page_count=parsed_doc.metadata.page_count,
        word_count=parsed_doc.metadata.total_word_count,
        status="uploaded",
        metadata_json=parsed_doc.metadata.extra_metadata,
    )
    db.add(doc_model)
    await db.flush()  # populate doc_model.id

    # Create DocumentPage DB Models
    for page in parsed_doc.pages:
        page_model = DocumentPage(
            document_id=doc_model.id,
            page_number=page.page_number,
            raw_text=page.raw_text,
            cleaned_text=None,
            word_count=page.word_count,
        )
        db.add(page_model)

    await db.commit()
    await db.refresh(doc_model)
    logger.info(f"Successfully uploaded & parsed document ID {doc_model.id}: {filename}")
    return doc_model


@router.get("", response_model=List[DocumentRead])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """
    List all uploaded documents.
    """
    result = await db.execute(select(Document).offset(skip).limit(limit).order_by(Document.id.desc()))
    documents = result.scalars().all()
    return documents


@router.get("/{document_id}", response_model=DocumentDetailRead)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed document information including extracted page texts.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages_result = await db.execute(
        select(DocumentPage).where(DocumentPage.document_id == document_id).order_by(DocumentPage.page_number.asc())
    )
    pages = pages_result.scalars().all()

    page_reads = [DocumentPageRead.model_validate(p) for p in pages]
    return DocumentDetailRead(
        id=doc.id,
        filename=doc.filename,
        title=doc.title,
        author=doc.author,
        file_type=doc.file_type,
        file_path=doc.file_path,
        file_size=doc.file_size,
        page_count=doc.page_count,
        word_count=doc.word_count,
        status=doc.status,
        metadata_json=doc.metadata_json,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        pages=page_reads,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete document, its associated page database records, and local file storage.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove local file
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception as e:
            logger.warning(f"Could not remove local file {doc.file_path}: {e}")

    await db.delete(doc)
    await db.commit()
    return None


@router.post("/{document_id}/process")
async def process_document(
    document_id: int,
    config: TextPreprocessingConfig = TextPreprocessingConfig(),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger preprocessing pipeline on uploaded document.
    Runs text normalization/tokenization for text/PDF/DOCX, or EDA stats for CSV datasets.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.file_type == "csv":
        eda_res = csv_processor.analyze_csv(doc.file_path, document_id=doc.id)
        doc.status = "processed"
        await db.commit()
        return eda_res.model_dump()

    # For text documents (PDF, DOCX, TXT)
    pages_result = await db.execute(
        select(DocumentPage).where(DocumentPage.document_id == document_id).order_by(DocumentPage.page_number.asc())
    )
    pages = pages_result.scalars().all()

    processed_pages = []
    total_processed_words = 0

    for page in pages:
        res = text_processor.process_text(
            raw_text=page.raw_text,
            document_id=doc.id,
            page_number=page.page_number,
            config=config,
        )
        page.cleaned_text = res.cleaned_text
        total_processed_words += res.word_count
        processed_pages.append(res.model_dump())

    doc.status = "processed"
    await db.commit()

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": "processed",
        "processed_pages": processed_pages,
    }

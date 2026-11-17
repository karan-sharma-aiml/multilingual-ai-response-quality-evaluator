"""Batch evaluation endpoint (CSV upload)."""

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_batch_service
from app.core.config import Settings, get_settings
from app.core.exceptions import BatchFileError, PayloadTooLargeError
from app.schemas.api import BatchEvaluationResponse, ErrorResponse
from app.services.batch_service import BatchEvaluationService

router = APIRouter(tags=["Batch"])

CSV_EXTENSION: str = ".csv"


@router.post(
    "/batch-evaluate",
    response_model=BatchEvaluationResponse,
    summary="Evaluate every row of a CSV file",
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
def batch_evaluate(
    file: UploadFile = File(..., description="CSV with columns: question, response, language "
                            "(optional: reference_answer, category, id)"),
    use_gemini: bool = Form(False, description="Also score every row with Gemini."),
    service: BatchEvaluationService = Depends(get_batch_service),
    settings: Settings = Depends(get_settings),
) -> BatchEvaluationResponse:
    """Evaluate and store each row. Rows that fail validation are reported in
    ``failed_records`` and do not stop the batch."""
    if not (file.filename or "").lower().endswith(CSV_EXTENSION):
        raise BatchFileError("Only .csv files are supported.")
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"The file exceeds the {settings.max_upload_bytes} byte upload limit."
        )
    return service.process_csv(content, use_gemini=use_gemini)

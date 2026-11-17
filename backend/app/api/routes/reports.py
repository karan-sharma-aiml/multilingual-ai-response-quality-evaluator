"""Report export endpoints (CSV, Excel, PDF)."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.api.dependencies import get_report_service
from app.api.filters import get_result_filters
from app.repositories import ResultFilters
from app.schemas.api import ErrorResponse
from app.services.report_service import (
    CSV_MEDIA_TYPE,
    EXCEL_MEDIA_TYPE,
    PDF_MEDIA_TYPE,
    ReportService,
)

router = APIRouter(prefix="/reports", tags=["Reports"])

_LIMIT = Query(None, ge=1, description="Maximum number of rows (newest first).")


def _download(content: bytes, media_type: str, extension: str) -> Response:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="evaluation_report_{stamp}.{extension}"'},
    )


@router.get("/csv", summary="Export evaluations as CSV", responses={422: {"model": ErrorResponse}})
def export_csv(
    filters: ResultFilters = Depends(get_result_filters),
    limit: int | None = _LIMIT,
    service: ReportService = Depends(get_report_service),
) -> Response:
    return _download(service.generate_csv(filters, limit), CSV_MEDIA_TYPE, "csv")


@router.get("/excel", summary="Export evaluations as an Excel workbook")
def export_excel(
    filters: ResultFilters = Depends(get_result_filters),
    limit: int | None = _LIMIT,
    service: ReportService = Depends(get_report_service),
) -> Response:
    return _download(service.generate_excel(filters, limit), EXCEL_MEDIA_TYPE, "xlsx")


@router.get("/pdf", summary="Export evaluations as a PDF report")
def export_pdf(
    filters: ResultFilters = Depends(get_result_filters),
    limit: int | None = _LIMIT,
    service: ReportService = Depends(get_report_service),
) -> Response:
    return _download(service.generate_pdf(filters, limit), PDF_MEDIA_TYPE, "pdf")

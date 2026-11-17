"""CSV, Excel and PDF report generation."""

import io
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.repositories import ResultFilters
from app.schemas.api import EvaluationResponse
from app.services.database_service import DatabaseService
from app.utils.sanitize import escape_spreadsheet_formula

logger = get_logger("reports")

CSV_MEDIA_TYPE: str = "text/csv; charset=utf-8"
EXCEL_MEDIA_TYPE: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PDF_MEDIA_TYPE: str = "application/pdf"

TEXT_COLUMNS: tuple[str, ...] = (
    "Language", "Category", "Question", "Response", "Method", "Gemini Reasoning", "Warnings",
)
SCORE_COLUMNS: tuple[str, ...] = (
    "Relevance", "Correctness", "Clarity", "Safety", "Hallucination Risk", "Duplicate %", "Overall Score",
)
REPORT_COLUMNS: tuple[str, ...] = (
    "ID", "Timestamp (UTC)", "Language", "Category", "Question", "Response",
    *SCORE_COLUMNS, "Method",
    "Gemini Relevance", "Gemini Correctness", "Gemini Clarity", "Gemini Safety",
    "Gemini Reasoning", "Warnings",
)
WRAPPED_COLUMNS: frozenset[str] = frozenset({"Question", "Response", "Gemini Reasoning", "Warnings"})
MAX_COLUMN_WIDTH: int = 60
MIN_COLUMN_WIDTH: int = 10
PDF_QUESTION_CHARS: int = 140
HEADER_COLOR: str = "1F3864"
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S"
FONT_NAME: str = "ReportFont"
FALLBACK_FONT: str = "Helvetica"
FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class PdfFont:
    """Resolves a TrueType font and replaces glyphs it cannot draw."""

    def __init__(self, configured_path: str | None = None) -> None:
        self.name = FALLBACK_FONT
        self._glyphs: set[int] | None = None
        candidates = [configured_path] if configured_path else []
        for path in (*candidates, *FONT_CANDIDATES):
            if path and Path(path).is_file():
                try:
                    font = TTFont(FONT_NAME, path)
                    pdfmetrics.registerFont(font)
                except Exception:  # noqa: BLE001 - try the next candidate
                    continue
                self.name = FONT_NAME
                self._glyphs = set(font.face.charToGlyph)
                break

    def safe(self, text: str) -> str:
        """Return ``text`` with undrawable characters replaced by ``?``."""
        if self._glyphs is None:
            return text.encode("cp1252", errors="replace").decode("cp1252")
        return "".join(ch if ord(ch) in self._glyphs or ch.isspace() else "?" for ch in text)


class ReportService:
    """Builds downloadable evaluation reports."""

    def __init__(self, database: DatabaseService, settings: Settings | None = None) -> None:
        self._database = database
        self._settings = settings or get_settings()

    # ------------------------------------------------------------- data
    def collect(self, filters: ResultFilters, limit: int | None = None) -> list[EvaluationResponse]:
        cap = self._settings.report_max_rows
        rows = self._database.export_results(filters, min(limit, cap) if limit else cap)
        self._database.log_event("report_generated", f"rows={len(rows)}")
        return rows

    @staticmethod
    def build_frame(rows: list[EvaluationResponse]) -> pd.DataFrame:
        """Tabular view; text cells are protected against formula injection."""
        records = []
        for row in rows:
            scores = row.gemini.scores or {}
            records.append(
                {
                    "ID": row.id,
                    "Timestamp (UTC)": _utc(row.created_at).strftime(TIMESTAMP_FORMAT),
                    "Language": row.language_name,
                    "Category": row.category or "",
                    "Question": row.question,
                    "Response": row.response,
                    "Relevance": row.relevance_score,
                    "Correctness": row.correctness_score,
                    "Clarity": row.clarity_score,
                    "Safety": row.safety_score,
                    "Hallucination Risk": row.hallucination_score,
                    "Duplicate %": row.duplicate_percentage,
                    "Overall Score": row.overall_score,
                    "Method": row.evaluation_method.value,
                    "Gemini Relevance": scores.get("relevance"),
                    "Gemini Correctness": scores.get("correctness"),
                    "Gemini Clarity": scores.get("clarity"),
                    "Gemini Safety": scores.get("safety"),
                    "Gemini Reasoning": row.gemini.reasoning or "",
                    "Warnings": "; ".join(row.warnings),
                }
            )
        frame = pd.DataFrame.from_records(records, columns=list(REPORT_COLUMNS))
        for column in TEXT_COLUMNS:
            frame[column] = frame[column].map(
                lambda value: escape_spreadsheet_formula(ILLEGAL_CHARACTERS_RE.sub("", str(value)))
            )
        return frame

    @staticmethod
    def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
        """Key figures shown on the Excel summary sheet and the PDF cover."""
        items: list[tuple[str, object]] = [("Total evaluations", len(frame))]
        for column in SCORE_COLUMNS:
            items.append((f"Average {column.lower()}", round(float(frame[column].mean()), 2) if len(frame) else None))
        if len(frame):
            by_language = frame.groupby("Language")["Overall Score"].agg(["count", "mean"])
            for language, stats in by_language.sort_index().iterrows():
                items.append((f"{language}: evaluations / average overall", f"{int(stats['count'])} / {stats['mean']:.2f}"))
        return pd.DataFrame(items, columns=["Metric", "Value"])

    # -------------------------------------------------------------- CSV
    def generate_csv(self, filters: ResultFilters, limit: int | None = None) -> bytes:
        started = time.perf_counter()
        frame = self.build_frame(self.collect(filters, limit))
        content = ("\ufeff" + frame.to_csv(index=False)).encode("utf-8")  # BOM: Excel-friendly UTF-8
        self._log("csv", len(frame), started)
        return content

    # ------------------------------------------------------------ Excel
    def generate_excel(self, filters: ResultFilters, limit: int | None = None) -> bytes:
        started = time.perf_counter()
        frame = self.build_frame(self.collect(filters, limit))
        summary = self.build_summary(frame)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Evaluations", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            self._style_sheet(writer.sheets["Evaluations"], frame)
            self._style_sheet(writer.sheets["Summary"], summary)
            if len(frame):
                column = get_column_letter(list(frame.columns).index("Overall Score") + 1)
                writer.sheets["Evaluations"].conditional_formatting.add(
                    f"{column}2:{column}{len(frame) + 1}",
                    ColorScaleRule(
                        start_type="num", start_value=0, start_color="F8696B",
                        mid_type="num", mid_value=50, mid_color="FFEB84",
                        end_type="num", end_value=100, end_color="63BE7B",
                    ),
                )
        self._log("excel", len(frame), started)
        return buffer.getvalue()

    @staticmethod
    def _style_sheet(sheet, frame: pd.DataFrame) -> None:
        header_fill = PatternFill("solid", fgColor=HEADER_COLOR)
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.freeze_panes = "A2"
        if sheet.max_row > 1:
            sheet.auto_filter.ref = sheet.dimensions
        for index, name in enumerate(frame.columns, start=1):
            longest = max([len(str(name))] + [len(str(v)) for v in frame[name].head(200)])
            width = max(MIN_COLUMN_WIDTH, min(MAX_COLUMN_WIDTH, longest + 2))
            sheet.column_dimensions[get_column_letter(index)].width = width
            if name in WRAPPED_COLUMNS:
                for cell in sheet[get_column_letter(index)][1:]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")

    # -------------------------------------------------------------- PDF
    def generate_pdf(self, filters: ResultFilters, limit: int | None = None) -> bytes:
        started = time.perf_counter()
        cap = self._settings.report_pdf_max_rows
        rows = self.collect(filters, min(limit, cap) if limit else cap)
        frame = self.build_frame(rows)
        fonts = PdfFont(self._settings.report_font_path)

        styles = getSampleStyleSheet()
        title = ParagraphStyle("Title", parent=styles["Title"], fontName=fonts.name, fontSize=18)
        body = ParagraphStyle("Body", parent=styles["BodyText"], fontName=fonts.name, fontSize=9)
        cell = ParagraphStyle("Cell", parent=body, fontSize=7, leading=8.5)

        def para(text: object, style: ParagraphStyle = cell) -> Paragraph:
            return Paragraph(escape(fonts.safe(str(text))), style)

        story: list = [
            Paragraph("Multilingual AI Response Quality Report", title),
            para(f"Generated {datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)} UTC - {len(frame)} evaluation(s)", body),
            Spacer(1, 10),
        ]
        summary = self.build_summary(frame)
        summary_table = LongTable([[para(m, body), para(v, body)] for m, v in summary.itertuples(index=False)],
                                  colWidths=[260, 120], hAlign="LEFT")
        summary_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ]))
        story += [summary_table, Spacer(1, 14)]

        header = ["ID", "Timestamp", "Language", "Question", "Rel", "Cor", "Cla", "Saf", "Hal", "Dup%", "Overall", "Gemini R/C/Cl/S", "Method"]
        data: list[list] = [[para(h, ParagraphStyle("H", parent=cell, textColor=colors.white)) for h in header]]
        for _, row in frame.iterrows():
            gemini = [row[c] for c in ("Gemini Relevance", "Gemini Correctness", "Gemini Clarity", "Gemini Safety")]
            gemini_text = "/".join(f"{v:g}" for v in gemini) if not any(pd.isna(v) for v in gemini) else "-"
            question = str(row["Question"])
            if len(question) > PDF_QUESTION_CHARS:
                question = question[: PDF_QUESTION_CHARS - 1] + "…"
            data.append([
                para(row["ID"]), para(row["Timestamp (UTC)"]), para(row["Language"]), para(question),
                para(row["Relevance"]), para(row["Correctness"]), para(row["Clarity"]), para(row["Safety"]),
                para(row["Hallucination Risk"]), para(row["Duplicate %"]), para(row["Overall Score"]),
                para(gemini_text), para(row["Method"]),
            ])
        table = LongTable(data, repeatRows=1,
                          colWidths=[26, 66, 46, 196, 32, 32, 32, 32, 32, 32, 40, 110, 52])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(f"#{HEADER_COLOR}")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FA")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
        if len(frame) >= cap:
            story += [Spacer(1, 6), para(f"Showing the newest {cap} evaluations; use CSV or Excel for the full export.", body)]

        buffer = io.BytesIO()
        document = SimpleDocTemplate(
            buffer, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28,
            title="Multilingual AI Response Quality Report",
        )

        def footer(canvas, doc) -> None:
            canvas.setFont(fonts.name, 7)
            canvas.drawRightString(landscape(A4)[0] - 24, 14, f"Page {doc.page}")

        document.build(story, onFirstPage=footer, onLaterPages=footer)
        self._log("pdf", len(frame), started)
        return buffer.getvalue()

    @staticmethod
    def _log(kind: str, rows: int, started: float) -> None:
        logger.info("Report generated: type=%s rows=%d duration_ms=%.1f", kind, rows, (time.perf_counter() - started) * 1000)

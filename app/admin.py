from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from .models import Document, AnalysisResult, ScrapeJob, GeneratedRegister, ScrapeSource, PreviousLegalRegister


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("filename", "doc_type", "pages_count", "avg_confidence", "status", "created_at")
    list_filter = ("doc_type", "status", "created_at")
    search_fields = ("filename", "json_path", "source_pdf")
@admin.register(GeneratedRegister)
class GeneratedRegisterAdmin(admin.ModelAdmin):
    list_display = ("id", "kind", "filename", "size_kb", "has_pdf", "created_at", "download_link")
    list_filter = ("kind", "created_at")
    search_fields = ("file_path", "filename")

    def has_pdf(self, obj: GeneratedRegister) -> bool:
        return bool(obj.pdf_data)
    has_pdf.boolean = True  # type: ignore[attr-defined]
    has_pdf.short_description = "PDF in DB"  # type: ignore[attr-defined]

    def size_kb(self, obj: GeneratedRegister) -> str:
        try:
            return f"{(obj.file_size or 0) / 1024:.1f} KB"
        except Exception:
            return "0 KB"
    size_kb.short_description = "Size"  # type: ignore[attr-defined]

    def download_link(self, obj: GeneratedRegister):
        url = reverse('download_generated_register', args=[obj.pk])
        return format_html('<a href="{}">Download</a>', url)
    download_link.short_description = "Download"  # type: ignore[attr-defined]


@admin.register(ScrapeSource)
class ScrapeSourceAdmin(admin.ModelAdmin):
    list_display = ("name", "url", "created_at")
    search_fields = ("name", "url")


@admin.register(PreviousLegalRegister)
class PreviousLegalRegisterAdmin(admin.ModelAdmin):
    list_display = ("filename", "rows_count", "status", "created_at")
    search_fields = ("filename", "pdf_path", "json_path")

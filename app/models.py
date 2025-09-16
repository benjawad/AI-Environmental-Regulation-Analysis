from django.db import models


class Document(models.Model):
    """Parsed document metadata + light summary for admin/queries."""
    filename = models.CharField(max_length=512)
    doc_type = models.CharField(max_length=128, blank=True, default="")
    pages_count = models.PositiveIntegerField(default=0)
    avg_confidence = models.FloatField(default=0.0)
    json_path = models.CharField(max_length=1024, blank=True, default="")
    json_content = models.JSONField(blank=True, null=True)  # optional: full parsed JSON
    source_pdf = models.CharField(max_length=1024, blank=True, default="")
    status = models.CharField(max_length=32, blank=True, default="")  # parsed|cached|error
    error = models.TextField(blank=True, default="")
    # New: store path to a previously uploaded legal register PDF (if applicable)
    previous_legal_registers = models.CharField(max_length=1024, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover
        return self.filename or f"Document#{self.pk}"


class AnalysisResult(models.Model):
    """Stores a single analysis run output, including the strict 11-column structured table."""
    description = models.TextField(blank=True, default="")
    structured_data = models.JSONField(default=list, blank=True)
    raw_result = models.JSONField(blank=True, null=True)   # optional: raw LLM/heuristic output
    kb_snapshot = models.JSONField(blank=True, null=True)  # optional: KB or context snapshot
    rows_count = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=32, blank=True, default="")  # success|failed
    error = models.TextField(blank=True, default="")
    pdf_path = models.CharField(max_length=1024, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover
        return f"AnalysisResult#{self.pk} ({self.status or 'unknown'})"


class ScrapeJob(models.Model):
    """Stores each web scraping run and its outcomes."""
    base_url = models.CharField(max_length=1024)
    params = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=32, blank=True, default="pending")  # pending|running|success|failed
    error = models.TextField(blank=True, default="")
    stats = models.JSONField(blank=True, null=True)  # e.g., {pdf_links_count, downloaded_count}
    links = models.JSONField(blank=True, null=True)  # optional: discovered links
    files = models.JSONField(blank=True, null=True)  # optional: downloaded files metadata
    output_dir = models.CharField(max_length=1024, blank=True, default="")
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:  # pragma: no cover
        return f"ScrapeJob#{self.pk} {self.base_url} ({self.status})"


class GeneratedRegister(models.Model):
    """Tracks generated register files (legal and commitment)."""
    KIND_CHOICES = (
        ("legal", "Legal Register"),
        ("commitment", "Commitment Register"),
    )
    kind = models.CharField(max_length=32, choices=KIND_CHOICES)
    analysis = models.ForeignKey(AnalysisResult, on_delete=models.SET_NULL, null=True, blank=True)
    # Legacy path on disk (kept for backward compatibility)
    file_path = models.CharField(max_length=1024, blank=True, default="")
    file_size = models.BigIntegerField(default=0)
    sha256 = models.CharField(max_length=64, blank=True, default="")
    rows_count = models.PositiveIntegerField(default=0)
    # New: store the actual PDF bytes in DB
    filename = models.CharField(max_length=255, blank=True, default="")
    pdf_data = models.BinaryField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.kind} -> {self.file_path}"


class ScrapeSource(models.Model):
    """Predefined websites that end users can pick for web scraping."""
    name = models.CharField(max_length=255)
    url = models.URLField(max_length=1024, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name} ({self.url})"


class PreviousLegalRegister(models.Model):
    """Stores previously uploaded legal register PDFs and their parsed JSON outputs."""
    filename = models.CharField(max_length=512, blank=True, default="")
    pdf_path = models.CharField(max_length=1024, blank=True, default="")
    json_path = models.CharField(max_length=1024, blank=True, default="")
    rows_count = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=32, blank=True, default="uploaded")  # uploaded|parsed|error
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Previous legal register"
        verbose_name_plural = "Previous legal registers"

    def __str__(self) -> str:  # pragma: no cover
        return self.filename or f"PreviousLegalRegister#{self.pk}"

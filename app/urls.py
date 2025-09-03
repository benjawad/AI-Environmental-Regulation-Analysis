from django.conf import settings
from django.urls import path

from app.services.legal_register.views import analyze_legal_register_llm, dashboard, deps_report, download_json, download_legal_register_pdf, download_pdf, list_jsons, list_pdfs,  parse_dir, parse_file, parse_upload, start_scrape, list_files, download_file, cleanup_files, preview_html
from . import views



urlpatterns = [
    path('', views.home, name='home'),
    path("scrape/", views.scrape_view, name="scrape"),
    path("process_pdfs/", views.process_pdfs_view, name="process_pdfs"),
    path("status/", views.status_view, name="status"),
    path("parser/validate/", views.parser_validate_view, name="parser_validate"),
    path("analyze_commitments/", views.analyze_commitments_view, name="analyze_commitments"),
    path("download_commitment_register/", views.download_commitment_register_view, name="download_commitment_register"),

    
    path("bo_scrape/dashboard/", dashboard, name="dashboard"),
    path("bo_scrape/start/", start_scrape, name="start_scrape"),
    path("bo_scrape/list/", list_files, name="list_files"),
    path("bo_scrape/file/<str:filename>", download_file, name="download_file"),
    path("bo_scrape/cleanup/", cleanup_files, name="cleanup_files"),
    path("bo_scrape/preview/", preview_html, name="preview_html"),

    path("bo_parse/upload/", parse_upload, name="parse_upload"),
    path("bo_parse/parse/<str:filename>/", parse_file, name="parse_file"),
    path("bo_parse/parse_dir/", parse_dir, name="parse_dir"),
    path("bo_parse/deps/", deps_report, name="deps_report"),
    path("bo_parse/list_pdfs/", list_pdfs, name="list_pdfs"),
    path("bo_parse/list_jsons/", list_jsons, name="list_jsons"),
    path("bo_parse/pdf/<str:filename>/", download_pdf, name="download_pdf"),
    path("bo_parse/json/<str:filename>/", download_json, name="download_json"),
    path("llm/analyze/", analyze_legal_register_llm, name="llm_analyze"),
    path("llm/download_pdf/", download_legal_register_pdf, name="download_legal_register_pdf"),





] 


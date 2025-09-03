# 🌍 AI Environmental Regulation Analysis

**AI-powered toolkit for analyzing environmental regulations and compliance**

---

## 🔎 Overview

**AI Environmental Regulation Analysis** is a platform designed to streamline the review and interpretation of environmental regulatory texts.
It leverages **Large Language Models (LLMs)** and **Optical Character Recognition (OCR)** to help legal professionals, environmental consultants, and regulatory analysts **extract, summarize, and understand compliance obligations** from scanned or digital documents.

---

## 🚀 Key Features

- 📝 **OCR Integration** — Converts scanned PDFs and image-based documents into searchable text using Tesseract OCR.
- 🤖 **Automated Regulatory Extraction** — Detects relevant clauses and compliance requirements using AI/LLM models.
- 📌 **Summarization & Highlighting** — Generates concise summaries of long regulations, pinpointing critical obligations.
- 🌐 **Cross-Jurisdiction Comparison** — Compares policies across regions, highlighting subtle differences.
- 🖥 **Django-Based Application** — Simple and production-ready backend for document upload and analysis.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://www.djangoproject.com/m/img/logos/django-logo-negative.png" alt="Django" height="40"/>
  <img src="https://tesseract.projectnaptha.com/assets/img/tesseract.png" alt="Tesseract OCR" height="40"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/ee/Google_Gemini_logo.svg" alt="Gemini" height="40"/>
  <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" height="40"/>
</p>

---

## 📦 Badges

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Django](https://img.shields.io/badge/Django-Backend-green?logo=django)
![OCR](https://img.shields.io/badge/Tesseract-OCR-orange?logo=google)
![LLM](https://img.shields.io/badge/AI-LLM-yellow?logo=openai)

---

## ⚡ Getting Started

### 1. Prerequisites
- Python 3.8+
- Tesseract OCR installed and added to your system PATH

---

### 2. Tesseract OCR Installation

#### 🖥️ Windows
1. Download Tesseract OCR from the [UB Mannheim releases](https://github.com/UB-Mannheim/tesseract/wiki).
2. During installation, **check "Add to system PATH"**.
3. (Optional) Install additional language packs you need.
4. Verify installation:
   ```bash
   tesseract --version
5. Add an environment variable if needed:
Name: TESSDATA_PREFIX
Value: C:\Program Files\Tesseract-OCR\tessdata

#### 🐧 Linux (Debian/Ubuntu)
```
sudo apt update
sudo apt install tesseract-ocr -y
sudo apt install libtesseract-dev -y
tesseract --version
```
#### 🍏 macOS (with Homebrew)
```
brew install tesseract
tesseract --version
```

### 3. Project Installation
```git clone [https://github.com/benjawad/AI-Environmental-Regulation-Analysis.git](https://github.com/benjawad/AI-Environmental-Regulation-Analysis.git)
cd AI-Environmental-Regulation-Analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Setup database
python manage.py migrate

# Run the development server
python manage.py runserver
```
### 4. Django Configuration for OCR
In your settings.py, add the path to the Tesseract executable (Windows only):
``` TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  ```
In your code (e.g., views.py):
```
import pytesseract
from PIL import Image
from django.conf import settings

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD_PATH
```

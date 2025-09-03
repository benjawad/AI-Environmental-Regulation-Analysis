# 🌍 AI Environmental Regulation Analysis

**AI-powered toolkit for analyzing environmental regulations and compliance**

---

## 🔎 Overview

**AI Environmental Regulation Analysis** is an intelligent platform designed to streamline the review and interpretation of environmental regulatory texts. It leverages AI technologies—including Large Language Models (LLMs) and optical character recognition (OCR)—to help legal professionals, environmental consultants, and regulatory analysts extract, summarize, and understand compliance obligations across diverse jurisdictions.

---

## 🚀 Key Features

- 📝 **OCR Integration** — Converts scanned PDFs and image-based documents into searchable, editable text.  
- 🤖 **Automated Regulatory Extraction** — Detects relevant clauses and compliance requirements using LLMs.  
- 📌 **Summarization & Highlighting** — Generates concise summaries of long regulations, pinpointing critical obligations.  
- 🌐 **Cross-Jurisdiction Comparison** — Compares environmental policies across regions, highlighting subtle differences.  
- 💻 **Web-Based Dashboard** — Upload documents, inspect AI insights, and export results via an intuitive UI.  

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://www.djangoproject.com/m/img/logos/django-logo-negative.png" alt="Django" height="40"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg" alt="React" height="40"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/Tesseract_OCR_logo_%28Google%29.png" alt="Tesseract OCR" height="40"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/GEMINI_Logo.svg" alt="GEMENI" height="40"/>
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

### Prerequisites
- Python 3.8+  
- Tesseract OCR installed and in system PATH  
- Node.js & npm/yarn (for frontend)  

### Installation

```bash
git clone https://github.com/benjawad/AI-Environmental-Regulation-Analysis.git
cd AI-Environmental-Regulation-Analysis
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Setup database
python manage.py migrate

# Run the development server
python manage.py runserver



# Tesseract OCR Integration with Django 
Tesseract OCR Integration with Django
This guide provides a step-by-step process for integrating Tesseract OCR into a Django project for a production environment on Windows. It covers the installation, configuration, and best practices for reliable text extraction from images.

1. Requirements
Tesseract OCR Engine: A standalone application that must be installed on the Windows server.

Python Libraries: pytesseract, Pillow.

Django Project: An existing Django application.

2. Step-by-Step Production Setup
Step 2.1: Install Tesseract on the Server
Download and run the Tesseract installer for Windows from a reliable source like the UB Mannheim GitHub repository. During installation:

Crucially, select "Add to system PATH for all users."

Select the language packs you need (e.g., eng for English) to download the necessary .traineddata files.

Verify the installation by opening a new command prompt and running tesseract --version.

Step 2.2: Configure System Environment Variables
To ensure Tesseract is accessible to your application, set the following system variables:

Open "Environment Variables" from the Windows Start Menu.

Under "System variables," verify that the Path variable includes the Tesseract installation directory (e.g., C:\Program Files\Tesseract-OCR).

Add a new system variable:

Variable name: TESSDATA_PREFIX

Variable value: The path to the tessdata folder, usually C:\Program Files\Tesseract-OCR\tessdata. This tells Tesseract where to find language data.

Restart any running command-line windows or your server to apply the changes.

Step 2.3: Set Up Django Project
Activate your project's virtual environment.

Install the required Python libraries:

Bash

pip install pytesseract
pip install Pillow
Update settings.py: Add the Tesseract executable path to your Django settings for robust configuration.

Python

## settings.py
import os

TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Configure pytesseract in your code: In your application logic (e.g., views.py), use the settings variable to specify the Tesseract command path. This is a best practice for production environments.

Python

## views.py
import pytesseract
from PIL import Image
from django.conf import settings

## Configure the pytesseract command path
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD_PATH
3. Production Best Practices
For a reliable and scalable production setup, consider the following:

Image Pre-processing: Use libraries like Pillow or OpenCV to improve image quality before OCR. Common techniques include:

Deskewing: Straightening tilted images.

Binarization: Converting images to black and white.

Noise Reduction: Removing visual artifacts.

Asynchronous Processing: OCR is a resource-intensive task. Use a task queue like Celery with a message broker (e.g., RabbitMQ or Redis) to run the OCR process in the background. This prevents your web server from blocking and ensures a smooth user experience.

Robust Error Handling: Implement extensive try...except blocks and logging to capture and handle any failures during the OCR process, file I/O, or image processing.

Security: Always validate uploaded files to prevent malicious uploads. Store them securely and delete temporary files after processing.

4. Troubleshooting
Tesseract is not installed or it's not in your PATH:

Ensure the Tesseract installer was run successfully and you selected the "Add to system PATH" option.

Verify the path in your settings.py is correct and matches the installation location.

Check that the TESSDATA_PREFIX environment variable is set correctly.

Could not open file:

Check that the user account running your web server (e.g., the IIS app pool user) has read and write permissions to the folders where images are stored and processed.








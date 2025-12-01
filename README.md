<<<<<<< HEAD
# 🔒 Universal Redaction Tool
## 🔒 Universal Redaction Tool

Comprehensive PII detection and redaction system combining Microsoft Presidio, spaCy NLP, and targeted regex recognizers. This repository provides an interactive Streamlit UI, batch processing utilities, and accuracy evaluation tools for safely removing or masking sensitive information from text and documents.

---

## Quick Overview

- Purpose: Detect and redact PII (names, emails, phones, IPs, credit cards, etc.) from text, documents, and structured data.
- Modes: Live redaction (interactive), Batch processing (files), Accuracy evaluation (compare to ground truth).

---

## **What the project uses**

- **Streamlit** — Web UI and user interaction (tabs, file upload, download).
- **Microsoft Presidio** — Analyzer and anonymizer engines (core PII detection & anonymization operators).
- **spaCy** — Underlying NLP model for contextual NER (uses `en_core_web_sm` or `en_core_web_lg`).
- **Custom Pattern Recognizers** — Regex-based recognizers added to Presidio for domain-specific tokens (credit cards, IPs, file names, API keys, etc.).
- **Pandas** — File/structured data handling (CSV/Excel/JSON processing).
- **pypdf / python-docx** — Document readers for PDF/DOCX extraction.
- **reportlab** — Optional PDF output generation for batch results.

---

## Entities Detected (default list)

This system uses Presidio built-ins plus custom regex recognizers. Notable entities the app targets:

- PERSON
- EMAIL_ADDRESS
- PHONE_NUMBER
- CREDIT_CARD (credit/debit card numbers)
- IP_ADDRESS (IPv4)
- URL
- DATE
- TIME
- LOCATION / GPE
- ZIP_CODE
- FILE_NAME
- API_KEY
- MAC_ADDRESS
- US_SSN
- PASSPORT_NUMBER
- EMPLOYEE_ID
- MEDICAL_ID
- STREET_ADDRESS

You can extend `logic.py::add_custom_recognizers()` to add or tweak recognizers.

---

## How it works (high level)

1. Input is provided via the Streamlit UI (live text) or uploaded files (CSV/XLSX/JSON/PDF/DOCX/TXT).
2. Text is extracted and passed to `logic.get_analyzer()` which configures Presidio with a spaCy NLP engine and extra regex recognizers.
3. `AnalyzerEngine.analyze()` returns entity spans with types and confidence scores.
4. Overlapping detections are resolved by priority rules in `logic.resolve_overlaps()`.
5. Redaction modes supported:
   - `redact` — Remove detected spans and preserve surrounding punctuation/spacing.
   - `mask` — Replace spans with bracketed tags (e.g., `[EMAIL_ADDRESS]`).
   - `tag_angular` — Insert angular tags (e.g., `<PERSON>`) for visualization.
6. For masking/anonymization, `presidio_anonymizer.AnonymizerEngine` is used with `OperatorConfig` map to produce final text.

---

## Installation (recommended)

1. Create & activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install project deps:
```bash
pip install -r requirements.txt
```
3. (If not included) download spaCy model you prefer:
```bash
python -m spacy download en_core_web_sm
# or for higher accuracy (larger):
python -m spacy download en_core_web_lg
```

---

## Running the app

- Local run (default):
```bash
streamlit run app.py
```

- In containers or remote hosts expose on all interfaces:
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Open the URL printed by Streamlit (e.g. `http://localhost:8501` or `http://0.0.0.0:8502`).

---

## Accuracy evaluation

- Use the **Accuracy Check** tab to paste Original, Redacted (system output), and Ground Truth (expected redaction). The app uses `accuracy.py` which performs token-based, sequence-alignment scoring (precision/recall/F1) and content-preservation checks.
- The accuracy logic is tolerant to token shifts and punctuation; it reports:
  - Total PII tokens expected
  - Correctly redacted tokens
  - Missed tokens
  - Precision / Recall / F1

Tips: Create a set of curated ground-truth examples covering your domains (emails, URLs, phone formats, labeled names with prefixes like `Name:`) to tune custom patterns.

---

## Customization & tuning

- Add regex-based recognizers in `logic.add_custom_recognizers()` for domain tokens (API keys, product SKUs, vendor-specific IDs).
- Adjust `priority` map in `logic.resolve_overlaps()` to prefer certain entity types when spans overlap.
- Change `USE_SMALL_MODEL` in `logic.py` to `True` to use `en_core_web_sm` for faster/demo runs.
- Modify anonymizer operators (`OperatorConfig`) to switch between `replace`, `mask`, `hash`, or custom logic.

Example: to add a strict company customer ID pattern, add a PatternRecognizer with a suitable regex and supported_entity name.

---

## Troubleshooting

- Import errors: make sure your virtualenv is active and `pip install -r requirements.txt` succeeded.
- Presidio errors: ensure `presidio-analyzer` and `presidio-anonymizer` are installed and compatible with your Python version.
- spaCy model: If detection seems weak, install `en_core_web_lg` and set `USE_SMALL_MODEL=False` in `logic.py`.
- Blank UI / port conflicts: if Streamlit reports port already in use, run on a different port or kill the process using that port.

---

## Performance & limitations

- Accuracy depends on the spaCy model and the quality of regex patterns. Hybrid approach minimizes misses but may produce false positives — tune patterns and priority weights accordingly.
- Large models (`en_core_web_lg`) improve NER recall for ambiguous names but require more RAM.
- This project performs in-memory processing; large files are memory bound.

---

## Contributing

- Improve patterns in `logic.py` and add test cases for tricky contexts (e.g., names following labels, sentence fragments like "on the evening of").
- Add more examples to the accuracy test suite and update `accuracy.py` if new tokenization behavior is needed.

---

## License

Add your license information here.

---

Last updated: November 29, 2025
=======
# 🔒 Universal Redaction Tool
## 🔒 Universal Redaction Tool

Comprehensive PII detection and redaction system combining Microsoft Presidio, spaCy NLP, and targeted regex recognizers. This repository provides an interactive Streamlit UI, batch processing utilities, and accuracy evaluation tools for safely removing or masking sensitive information from text and documents.

---

## Quick Overview

- Purpose: Detect and redact PII (names, emails, phones, IPs, credit cards, etc.) from text, documents, and structured data.
- Modes: Live redaction (interactive), Batch processing (files), Accuracy evaluation (compare to ground truth).

---

## **What the project uses**

- **Streamlit** — Web UI and user interaction (tabs, file upload, download).
- **Microsoft Presidio** — Analyzer and anonymizer engines (core PII detection & anonymization operators).
- **spaCy** — Underlying NLP model for contextual NER (uses `en_core_web_sm` or `en_core_web_lg`).
- **Custom Pattern Recognizers** — Regex-based recognizers added to Presidio for domain-specific tokens (credit cards, IPs, file names, API keys, etc.).
- **Pandas** — File/structured data handling (CSV/Excel/JSON processing).
- **pypdf / python-docx** — Document readers for PDF/DOCX extraction.
- **reportlab** — Optional PDF output generation for batch results.

---

## Entities Detected (default list)

This system uses Presidio built-ins plus custom regex recognizers. Notable entities the app targets:

- PERSON
- EMAIL_ADDRESS
- PHONE_NUMBER
- CREDIT_CARD (credit/debit card numbers)
- IP_ADDRESS (IPv4)
- URL
- DATE
- TIME
- LOCATION / GPE
- ZIP_CODE
- FILE_NAME
- API_KEY
- MAC_ADDRESS
- US_SSN
- PASSPORT_NUMBER
- EMPLOYEE_ID
- MEDICAL_ID
- STREET_ADDRESS

You can extend `logic.py::add_custom_recognizers()` to add or tweak recognizers.

---

## How it works (high level)

1. Input is provided via the Streamlit UI (live text) or uploaded files (CSV/XLSX/JSON/PDF/DOCX/TXT).
2. Text is extracted and passed to `logic.get_analyzer()` which configures Presidio with a spaCy NLP engine and extra regex recognizers.
3. `AnalyzerEngine.analyze()` returns entity spans with types and confidence scores.
4. Overlapping detections are resolved by priority rules in `logic.resolve_overlaps()`.
5. Redaction modes supported:
   - `redact` — Remove detected spans and preserve surrounding punctuation/spacing.
   - `mask` — Replace spans with bracketed tags (e.g., `[EMAIL_ADDRESS]`).
   - `tag_angular` — Insert angular tags (e.g., `<PERSON>`) for visualization.
6. For masking/anonymization, `presidio_anonymizer.AnonymizerEngine` is used with `OperatorConfig` map to produce final text.

---

## Installation (recommended)

1. Create & activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install project deps:
```bash
pip install -r requirements.txt
```
3. (If not included) download spaCy model you prefer:
```bash
python -m spacy download en_core_web_sm
# or for higher accuracy (larger):
python -m spacy download en_core_web_lg
```

---

## Running the app

- Local run (default):
```bash
streamlit run app.py
```

- In containers or remote hosts expose on all interfaces:
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Open the URL printed by Streamlit (e.g. `http://localhost:8501` or `http://0.0.0.0:8502`).

---

## Accuracy evaluation

- Use the **Accuracy Check** tab to paste Original, Redacted (system output), and Ground Truth (expected redaction). The app uses `accuracy.py` which performs token-based, sequence-alignment scoring (precision/recall/F1) and content-preservation checks.
- The accuracy logic is tolerant to token shifts and punctuation; it reports:
  - Total PII tokens expected
  - Correctly redacted tokens
  - Missed tokens
  - Precision / Recall / F1

Tips: Create a set of curated ground-truth examples covering your domains (emails, URLs, phone formats, labeled names with prefixes like `Name:`) to tune custom patterns.

---

## Customization & tuning

- Add regex-based recognizers in `logic.add_custom_recognizers()` for domain tokens (API keys, product SKUs, vendor-specific IDs).
- Adjust `priority` map in `logic.resolve_overlaps()` to prefer certain entity types when spans overlap.
- Change `USE_SMALL_MODEL` in `logic.py` to `True` to use `en_core_web_sm` for faster/demo runs.
- Modify anonymizer operators (`OperatorConfig`) to switch between `replace`, `mask`, `hash`, or custom logic.

Example: to add a strict company customer ID pattern, add a PatternRecognizer with a suitable regex and supported_entity name.

---

## Troubleshooting

- Import errors: make sure your virtualenv is active and `pip install -r requirements.txt` succeeded.
- Presidio errors: ensure `presidio-analyzer` and `presidio-anonymizer` are installed and compatible with your Python version.
- spaCy model: If detection seems weak, install `en_core_web_lg` and set `USE_SMALL_MODEL=False` in `logic.py`.
- Blank UI / port conflicts: if Streamlit reports port already in use, run on a different port or kill the process using that port.

---

## Performance & limitations

- Accuracy depends on the spaCy model and the quality of regex patterns. Hybrid approach minimizes misses but may produce false positives — tune patterns and priority weights accordingly.
- Large models (`en_core_web_lg`) improve NER recall for ambiguous names but require more RAM.
- This project performs in-memory processing; large files are memory bound.

---

## Contributing

- Improve patterns in `logic.py` and add test cases for tricky contexts (e.g., names following labels, sentence fragments like "on the evening of").
- Add more examples to the accuracy test suite and update `accuracy.py` if new tokenization behavior is needed.

---

## License

Add your license information here.

---

Last updated: November 29, 2025
>>>>>>> dae57f4 (feat: Implement Universal Redaction Tool with accuracy evaluation)

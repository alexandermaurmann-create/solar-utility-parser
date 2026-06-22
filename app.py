import os
import io
import json
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pdf2image import convert_from_path
import anthropic

app = Flask(__name__, static_folder="static")
CORS(app)

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

EXTRACT_PROMPT = """You are extracting data from a utility electricity bill.

Extract ONLY these fields and return a single JSON object — no markdown, no extra text:

{
  "billing_period_start": "MM/DD/YYYY or null",
  "billing_period_end": "MM/DD/YYYY or null",
  "on_peak_kwh": number or null,
  "mid_peak_kwh": number or null,
  "off_peak_kwh": number or null,
  "total_kwh": number or null,
  "delivery_charge": number or null,
  "regulatory_charge": number or null
}

Rules:
- on_peak_kwh = the kWh QUANTITY used during On-Peak / Highest Price period. This is the large number before "kWh" on that line (e.g. "508.091 kWh On-peak" → 508.091). Do NOT use the dollar amount on the same line.
- mid_peak_kwh = the kWh QUANTITY used during Mid-Peak / Mid Price period. Same rule — use the number before "kWh", not the dollar charge.
- off_peak_kwh = the kWh QUANTITY used during Off-Peak / Lowest Price period. Same rule.
- total_kwh = total kWh used for the billing period (from meter reading table if available, labeled "kWh Used")
- delivery_charge = Delivery charge in dollars (number only, no $ sign)
- regulatory_charge = Regulatory charge in dollars (number only, no $ sign)
- billing_period_start/end = meter reading period start and end dates
- If a field is not found, use null
- Return ONLY the JSON object, nothing else
"""

def pdf_to_images_b64(pdf_path):
    """Convert PDF pages to base64-encoded PNG images."""
    images = convert_from_path(pdf_path, dpi=200)
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.standard_b64encode(buf.getvalue()).decode("utf-8"))
    return result

def extract_with_claude(images_b64):
    """Send PDF page images to Claude and extract structured bill data."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Build content: all page images + the extraction prompt
    content = []
    for img_b64 in images_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64
            }
        })
    content.append({"type": "text", "text": EXTRACT_PROMPT})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)

def month_from_date(date_str):
    """Parse 'MM/DD/YYYY' and return (month_name, year)."""
    if not date_str:
        return None, None
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")
        return dt.strftime("%B"), dt.year
    except Exception:
        return None, None

def build_sorted_rows(bills):
    """Sort bills by calendar month (Jan→Dec), using most recent 12."""
    known = [b for b in bills if b.get("billing_month") and b["billing_month"] in MONTH_ORDER]
    unknown = [b for b in bills if b not in known]

    # Sort by year then month, take most recent 12
    known.sort(key=lambda b: (b["billing_year"], MONTH_ORDER.index(b["billing_month"])))
    recent = known[-12:]

    # Re-sort Jan→Dec for display
    recent.sort(key=lambda b: MONTH_ORDER.index(b["billing_month"]))
    return recent + unknown

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    bills = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue

        tmp_path = f"/tmp/{f.filename}"
        f.save(tmp_path)

        try:
            images_b64 = pdf_to_images_b64(tmp_path)
            data = extract_with_claude(images_b64)
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to process {f.filename}: {str(e)}"}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Determine billing month from end date
        month, year = month_from_date(data.get("billing_period_end"))

        bills.append({
            "filename": f.filename,
            "billing_month": month or "Unknown",
            "billing_year": year or 0,
            "period_start": data.get("billing_period_start") or "",
            "period_end": data.get("billing_period_end") or "",
            "on_peak_kwh": data.get("on_peak_kwh"),
            "mid_peak_kwh": data.get("mid_peak_kwh"),
            "off_peak_kwh": data.get("off_peak_kwh"),
            "total_kwh": data.get("total_kwh"),
            "delivery_charge": data.get("delivery_charge"),
            "regulatory_charge": data.get("regulatory_charge"),
        })

    if not bills:
        return jsonify({"error": "No valid PDF files found"}), 400

    rows = build_sorted_rows(bills)
    return jsonify({"rows": rows})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

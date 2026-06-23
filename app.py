import os
import io
import json
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import anthropic

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
CORS(app)

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

EXTRACT_PROMPT = """You are extracting data from a utility electricity bill.

First, identify the bill type:
- "TOU" = Time-of-Use: has On-peak, Mid-peak, Off-peak charges
- "ULO" = Ultra-Low Overnight: has On-peak, Mid-peak, Off-peak AND Overnight charges
- "Tiered" = Tiered Rate: has Lower Tier and Higher Tier charges (no on/mid/off peak)

Extract ONLY these fields and return a single JSON object — no markdown, no extra text:

{
  "bill_type": "TOU" or "ULO" or "Tiered",
  "billing_period_start": "MM/DD/YYYY or null",
  "billing_period_end": "MM/DD/YYYY or null",
  "on_peak_kwh": number or null,
  "mid_peak_kwh": number or null,
  "off_peak_kwh": number or null,
  "overnight_kwh": number or null,
  "tier1_kwh": number or null,
  "tier2_kwh": number or null,
  "total_kwh": number or null,
  "delivery_charge": number or null,
  "regulatory_charge": number or null,
  "monthly_usage_history": [
    {"date": "DD MMM YY", "kwh": number},
    ...
  ]
}

Rules:
- bill_type: detect from the electricity charges section. If it says "On-peak/Mid-peak/Off-peak" it's TOU or ULO. If it also has "Overnight" it's ULO. If it says "Lower Tier/Higher Tier" it's Tiered.
- on_peak_kwh = kWh QUANTITY for On-Peak / Highest Price (TOU and ULO only). Use the number BEFORE "kWh" on that line, NOT the dollar amount.
- mid_peak_kwh = kWh QUANTITY for Mid-Peak / Mid Price (TOU and ULO only). Same rule.
- off_peak_kwh = kWh QUANTITY for Off-Peak / Lowest Price (TOU and ULO only). Same rule.
- overnight_kwh = kWh QUANTITY for Overnight period (ULO only). Ignore any negative/credit entries — only use positive consumption values.
- tier1_kwh = SUM of ALL Lower Tier kWh entries (Tiered only). There may be 2 rows if billing crossed a rate change date — add them together.
- tier2_kwh = SUM of ALL Higher Tier kWh entries (Tiered only). Same — add all Higher Tier rows together.
- total_kwh = total kWh used for the billing period (from meter reading table, labeled "kWh Used")
- delivery_charge = Delivery charge in dollars (number only, no $ sign)
- regulatory_charge = Regulatory charge in dollars (number only, no $ sign)
- billing_period_start/end = meter reading period start and end dates
- monthly_usage_history = ALL rows from the "Compare Your Daily Usage" bar chart/table (typically 13-15 months). Format dates as "DD MMM YY" (e.g. "20 MAR 26"). kWh values as plain integers.
- For fields that don't apply to the detected bill type, use null.
- ALL numeric values must be plain numbers — NO commas, NO dollar signs, NO units.
- If a field is not found, use null. For monthly_usage_history use [] if not found.

IMPORTANT — digit accuracy:
- Read every digit carefully. Common mistakes: 6 vs 8, 1 vs 7, 5 vs 6, 0 vs 8, 3 vs 8.
- Monthly kWh values for a home are typically between 500–5000 — verify values make sense.
- Return ONLY the JSON object, nothing else
"""

VERIFY_PROMPT = """You previously extracted this data from a utility bill image:

{data}

Please re-read the image carefully and verify every number. Pay special attention to digits that look similar: 6 vs 8, 1 vs 7, 5 vs 6, 0 vs 8.

Return the corrected JSON object with the same structure. If a value was correct, keep it. If you spot an error, fix it. Return ONLY the JSON object, no other text.
"""

def enhance_image(img):
    """Boost contrast and sharpness to improve OCR accuracy."""
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)
    # Boost contrast
    img = ImageEnhance.Contrast(img).enhance(1.8)
    # Boost sharpness
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    # Slight brightness boost for dark/faded images
    img = ImageEnhance.Brightness(img).enhance(1.1)
    return img

def file_to_images_b64(file_path):
    """Convert a PDF or PNG file to a list of enhanced base64-encoded PNG images."""
    if file_path.lower().endswith(".png"):
        img = Image.open(file_path)
        img = enhance_image(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return [base64.standard_b64encode(buf.getvalue()).decode("utf-8")]
    else:
        images = convert_from_path(file_path, dpi=300)
        result = []
        for img in images:
            img = enhance_image(img)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result.append(base64.standard_b64encode(buf.getvalue()).decode("utf-8"))
        return result

def clean_json(raw):
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)

def extract_with_claude(images_b64):
    """Send images to Claude, extract data, then do a verification pass."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    image_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64}
        }
        for b64 in images_b64
    ]

    # Pass 1: initial extraction
    response1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": image_content + [{"type": "text", "text": EXTRACT_PROMPT}]}]
    )
    data = clean_json(response1.content[0].text)

    # Pass 2: verification — re-read the image and check every number
    verify_prompt = VERIFY_PROMPT.format(data=json.dumps(data, indent=2))
    response2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": image_content + [{"type": "text", "text": verify_prompt}]}]
    )
    verified = clean_json(response2.content[0].text)

    return verified

def parse_history_date(date_str):
    """Parse '20 MAR 26' or '20 MAR 2026' → (year, month_index 0-11)."""
    for fmt in ("%d %b %y", "%d %b %Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.year, dt.month - 1  # month_index 0=Jan
        except ValueError:
            continue
    return None, None

def build_monthly_history(raw_history):
    """
    Takes list of {date, kwh}, keeps most recent 12, returns sorted Jan→Dec.
    Each output row: {month_abbr, month_index, kwh}
    """
    if not raw_history:
        return []

    parsed = []
    for entry in raw_history:
        year, month_idx = parse_history_date(entry.get("date", ""))
        if year is not None:
            parsed.append({
                "year": year,
                "month_index": month_idx,
                "month_abbr": MONTH_ABBR[month_idx],
                "kwh": entry.get("kwh")
            })

    # Sort chronologically, take most recent 12
    parsed.sort(key=lambda x: (x["year"], x["month_index"]))
    recent = parsed[-12:]

    # Re-sort Jan→Dec by month index
    recent.sort(key=lambda x: x["month_index"])
    return recent

def month_from_date(date_str):
    """Parse 'MM/DD/YYYY' and return (month_name, year)."""
    if not date_str:
        return None, None
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")
        return dt.strftime("%B"), dt.year
    except Exception:
        return None, None

def build_sorted_rows(bills):
    """Sort bills by calendar month (Jan→Dec), using most recent 12."""
    known = [b for b in bills if b.get("billing_month") and b["billing_month"] in MONTH_ORDER]
    unknown = [b for b in bills if b not in known]

    known.sort(key=lambda b: (b["billing_year"], MONTH_ORDER.index(b["billing_month"])))
    recent = known[-12:]
    recent.sort(key=lambda b: MONTH_ORDER.index(b["billing_month"]))
    return recent + unknown

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Login — Solar Utility Parser</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #f0f4f8; display: flex; align-items: center;
           justify-content: center; min-height: 100vh; }
    .card { background: white; border-radius: 12px; padding: 2.5rem 2rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1); width: 100%; max-width: 360px; }
    h1 { font-size: 1.4rem; text-align: center; margin-bottom: 0.3rem; color: #2d3748; }
    p { text-align: center; color: #718096; font-size: 0.9rem; margin-bottom: 1.8rem; }
    input { width: 100%; padding: 0.75rem 1rem; border: 1px solid #e2e8f0;
            border-radius: 8px; font-size: 1rem; margin-bottom: 1rem; outline: none; }
    input:focus { border-color: #f6a623; }
    button { width: 100%; padding: 0.75rem; background: #f6a623; color: white;
             border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; }
    button:hover { background: #e09410; }
    .error { color: #c53030; font-size: 0.88rem; text-align: center; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>☀️ Solar Utility Parser</h1>
    <p>Enter the password to continue</p>
    {error}
    <form method="POST" action="/login">
      <input type="password" name="password" placeholder="Password" autofocus />
      <button type="submit">Sign In</button>
    </form>
  </div>
</body>
</html>"""

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == os.environ.get("APP_PASSWORD", "solar123"):
            session["authenticated"] = True
            return redirect(url_for("index"))
        return LOGIN_HTML.replace("{error}", '<p class="error">Incorrect password</p>'), 401
    return LOGIN_HTML.replace("{error}", "")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

def require_auth():
    return not session.get("authenticated")

@app.route("/")
def index():
    if require_auth():
        return redirect(url_for("login"))
    return send_from_directory("static", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    bills = []

    for f in files:
        ext = f.filename.lower().split(".")[-1]
        if ext not in ("pdf", "png"):
            continue

        tmp_path = f"/tmp/{f.filename}"
        f.save(tmp_path)

        try:
            images_b64 = file_to_images_b64(tmp_path)
            data = extract_with_claude(images_b64)
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to process {f.filename}: {str(e)}"}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        month, year = month_from_date(data.get("billing_period_end"))
        monthly_history = build_monthly_history(data.get("monthly_usage_history") or [])

        bills.append({
            "filename": f.filename,
            "bill_type": data.get("bill_type", "TOU"),
            "billing_month": month or "Unknown",
            "billing_year": year or 0,
            "period_start": data.get("billing_period_start") or "",
            "period_end": data.get("billing_period_end") or "",
            "on_peak_kwh": data.get("on_peak_kwh"),
            "mid_peak_kwh": data.get("mid_peak_kwh"),
            "off_peak_kwh": data.get("off_peak_kwh"),
            "overnight_kwh": data.get("overnight_kwh"),
            "tier1_kwh": data.get("tier1_kwh"),
            "tier2_kwh": data.get("tier2_kwh"),
            "total_kwh": data.get("total_kwh"),
            "delivery_charge": data.get("delivery_charge"),
            "regulatory_charge": data.get("regulatory_charge"),
            "monthly_history": monthly_history,
        })

    if not bills:
        return jsonify({"error": "No valid PDF or PNG files found"}), 400

    rows = build_sorted_rows(bills)
    return jsonify({"rows": rows})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

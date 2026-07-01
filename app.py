import os
import io
import re
import gc
import json
import base64
import uuid
import concurrent.futures
from collections import Counter
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import anthropic

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
CORS(app)

# --- Model routing (speed/accuracy balance) ---
# Detection is an easy structured task -> Haiku (fast/cheap).
# Number-reading extraction -> Sonnet (strong OCR, much faster than Opus).
# Verify is a second-opinion re-read; only run when pass-1 looks inconsistent.
META_MODEL = "claude-haiku-4-5-20251001"
EXTRACT_MODEL = "claude-sonnet-4-6"
VERIFY_MODEL = "claude-sonnet-4-6"

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
  "bill_date": "MM/DD/YYYY or null",
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
- bill_date = the bill's statement / invoice / bill date (look for "Statement", "Invoice Date", "Bill Date", or "Bill Print Date"). Used to label the month when no billing period range is shown.
- monthly_usage_history = EVERY row of the usage-history list (Toronto Hydro's "Compare Your Daily Usage" / "Read Date" list, typically 12-15 entries). For EACH row return:
  * "kwh" = the kWh value EXACTLY as printed next to that row — do NOT multiply, convert, or adjust it. If BOTH a total-kWh column and a per-day column are printed, use the TOTAL-kWh value here (e.g. read 1401, not 43.78).
  * "date" = each row's date (often the meter "Read Date"), formatted "DD MMM YY" (e.g. "26 FEB 26"). ALWAYS include a year. If only a month name is shown, infer each row's year from the list's date range and output e.g. "01 APR 25". Never return a bare month with no year.
  * Read every bar — do not skip any, and keep them in the chart's order.
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

HISTORY_PROMPT = """This utility bill shows a usage-history list/chart where each row has a DATE (often a meter "Read Date") and a kWh value printed next to it.

Transcribe EVERY row exactly as printed — do not measure bar heights, do not round, do not skip duplicates. If two rows share the same month (e.g. two reads in October), include BOTH as separate entries.

Return ONLY this JSON object, no markdown, no other text:
{"history": [{"date": "DD MMM YY", "kwh": number}, ...]}

Rules:
- The kWh value is the PRINTED NUMBER shown for that row (e.g. in a "kWh Usage" column). READ THE DIGITS. Do NOT estimate it from the length of any bar.
- "date": the row's date, formatted "DD MMM YY" with the year (e.g. "31 OCT 25"). If only a month is shown, use day 01 and infer the year from the chart's range.
- "kwh": the kWh value printed for that row, EXACTLY as shown (strip commas; e.g. "1,903" -> 1903). Read each digit carefully — distinguish 6/8, 1/7, 3/8, 5/6, 0/8.
- Include all rows, newest or oldest order is fine.
"""

HISTORY_BBOX_PROMPT = """Find the USAGE-HISTORY chart on this utility bill: a tall list of roughly 12-15 rows, each row a DATE (often a meter "Read Date") with a kWh value and usually a small horizontal bar. It is often titled something like "Compare Your Daily Usage" or "Usage History".

Do NOT return:
- the single current meter-reading row / billing summary (one row, e.g. "kWh Used 1438") — usually near the bottom of the bill,
- a small grouped bar chart comparing this period vs last year (e.g. "Time-of-Use Comparison"),
- the electricity charges breakdown.

Return ONLY the bounding box of that MANY-ROW history list as fractions of the page (0.0-1.0), tight around all its rows and including both the date column and the kWh-value column:
{"top": 0.0-1.0, "bottom": 0.0-1.0, "left": 0.0-1.0, "right": 0.0-1.0, "row_count": <approx number of rows>}

If there is no such multi-row history list, return {}. No markdown, no other text.
"""

# Fallback crop boxes (page fractions) for the usage-history table, by issuer
# substring. Used only when the model's locate step fails or is rejected. Toronto
# Hydro's "Compare Your Daily Usage" list sits in the top-right of the bill.
HISTORY_TABLE_BOXES = {
    "toronto hydro": {"top": 0.0, "bottom": 0.42, "left": 0.55, "right": 1.0},
}

CHART_META_PROMPT = """Analyze the usage history bar chart in this utility bill image.

Return ONLY this JSON — no markdown, no extra text:
{
  "has_bar_chart": true or false,
  "has_printed_numbers": true or false,
  "y_axis_max": number,
  "y_axis_min": 0,
  "y_axis_gridlines": [5000, 4000, 3000, 2000, 1000, 0],
  "bar_color_rgb": [R, G, B],
  "chart_top_pct": 0.0-1.0,
  "chart_bottom_pct": 0.0-1.0,
  "chart_left_pct": 0.0-1.0,
  "chart_right_pct": 0.0-1.0,
  "page_index": 0,
  "month_labels": ["Mar 25", "Apr 25", ...],
  "bar_count": 12,
  "bar_centers_pct": [0.04, 0.11, 0.19, ...],
  "chart_total_kwh": number or null,
  "current_period_kwh": number or null,
  "issuer": "utility company name printed on the bill, lowercase (e.g. toronto hydro), or null",
  "hst_number": "the HST / tax registration number printed on the bill (e.g. '86360 3726 RT0001'), or null"
}

Definitions:
- has_printed_numbers: true ONLY if each bar has its kWh value printed on or above it
- y_axis_max: the HIGHEST value actually labeled on the y-axis. Must equal the first entry of y_axis_gridlines. Do NOT invent values above what is printed.
- y_axis_gridlines: ONLY values that are actually labeled on the y-axis, from TOP to BOTTOM. e.g. if labels read 700, 600, 500, 400, 300, 200, 100, 0 return [700, 600, 500, 400, 300, 200, 100, 0]. Do NOT add values that are not labeled. Count the printed labels carefully — return EXACTLY that many entries, no more.
- bar_color_rgb: approximate RGB of the primary bar color, e.g. [140, 100, 190] for purple
- chart_*_pct: bar chart PLOT area (inside the axis lines) as fraction of image dimensions
- page_index: 0-indexed page number the chart is on
- month_labels: bar labels left to right, format "MMM YY". ONLY include months that have an actual visible bar — count the bars carefully
- bar_count: exact number of bars visible in the chart (count them). Must equal len(month_labels).
- bar_centers_pct: x-center of EACH bar as a fraction of the chart plot width (0.0=left edge, 1.0=right edge).
  Must have exactly the same number of entries as month_labels.
  Example for 4 bars evenly spaced with padding: [0.10, 0.35, 0.60, 0.85]
- chart_total_kwh: total kWh value printed near the chart (annual/period total), or null if not shown
- current_period_kwh: the TOTAL kWh used in the current billing period — the value the newest (most recent) history bar represents. Read it from a "Current month" row near the chart if shown, OR compute it from the current bill's electricity usage (e.g. sum the TOU On/Mid/Off-peak kWh, or Tiered Base+Remaining kWh, or the meter "kWh used"). null only if no usage figure appears anywhere.
- issuer: the electricity utility that issued the bill, read from the logo/header (e.g. "toronto hydro"). Lowercase. null if unclear. (Some bills omit the brand name — that's fine, return null and rely on hst_number.)
- hst_number: the HST / GST tax registration number printed on the bill, usually near "HST" or "Reg. no." (e.g. "86360 3726 RT0001"). This identifies the utility even when the brand name is absent. null if not found.

If there is no bar chart, return {"has_bar_chart": false}.
"""


def enhance_image(img):
    """Boost contrast and sharpness to improve OCR accuracy."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    return img


def _sniff_file_kind(file_path):
    """Identify a file by its MAGIC BYTES, not its extension (uploads are often
    mislabeled — a .png that's really a HEIC photo, a renamed PDF, etc.).
    Returns one of: 'pdf', 'image', 'heic', 'unknown'."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(32)
    except Exception:
        return "unknown"
    if head[:4] == b"%PDF":
        return "pdf"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if head[:3] == b"\xff\xd8\xff":                      # JPEG
        return "image"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if head[:2] == b"BM":                                # BMP
        return "image"
    if head[:4] in (b"II*\x00", b"MM\x00*"):             # TIFF
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head[4:8] == b"ftyp":                             # ISO-BMFF: HEIC/HEIF/AVIF
        return "heic"
    return "unknown"


def _render_one(original):
    """Shared: from a PIL page/image, return (original_rgb, enhanced_b64)."""
    original = original.convert("RGB")
    enhanced = enhance_image(original.copy())
    buf = io.BytesIO()
    enhanced.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return original, b64


def file_to_images(file_path):
    """
    Convert a PDF or image (PNG/JPEG/HEIC/…) to (pil_images, b64_strings).
    pil_images = original PIL images (for pixel analysis).
    b64_strings = enhanced base64 strings (for Claude OCR).
    Detection is by file content, so a mislabeled extension still works; an
    unreadable file raises a clear ValueError shown to the user.
    """
    pil_images = []
    b64_strings = []
    name = os.path.basename(file_path).split("_", 1)[-1]   # strip the uuid prefix
    kind = _sniff_file_kind(file_path)

    if kind == "pdf":
        # 150 DPI keeps plenty of detail for both Claude OCR (which downsamples to
        # ~1568px anyway) and pixel bar analysis, while using ~1/4 the memory of
        # 300 DPI — the 300 DPI render was OOM-killing the 512MB Render instance.
        pages = convert_from_path(file_path, dpi=150)
        for page in pages:
            original, b64 = _render_one(page)
            pil_images.append(original)
            b64_strings.append(b64)
        del pages
        return pil_images, b64_strings

    if kind == "heic":
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            raise ValueError(
                f"'{name}' looks like a HEIC/HEIF image (common from iPhone or Mac "
                "screenshots). Please re-save or export it as PNG or JPEG and upload again."
            )

    if kind in ("image", "heic"):
        try:
            with Image.open(file_path) as im:
                original, b64 = _render_one(im)
        except Exception:
            raise ValueError(
                f"Couldn't read '{name}' as an image. The file may be corrupted or in an "
                "unsupported format — please re-save it as PNG or JPEG and try again."
            )
        pil_images.append(original)
        b64_strings.append(b64)
        return pil_images, b64_strings

    raise ValueError(
        f"'{name}' isn't a readable PDF or image. Please upload a PDF, PNG, or JPEG "
        "(if it's a HEIC photo or a screenshot, export it as PNG first)."
    )


def clean_json(raw):
    """Strip markdown fences and extract the JSON object even if surrounded by prose."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    # Find outermost { ... } block, ignoring any prose before/after
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _hist_key(date_str):
    """Normalize a history-row date to (year, month_index, day) so the same row
    can be matched across independent transcriptions. day defaults to 15 for
    month-only labels (which never collide within a month)."""
    y, m = parse_history_date(date_str)
    if y is None:
        return None
    dm = re.match(r"\s*(\d{1,2})[\s\-/](?=[A-Za-z])", str(date_str))
    day = int(dm.group(1)) if dm else 15
    return (y, m, day)


def locate_history_table(client, image_content):
    """Ask the model for the bounding box (page fractions) of the date+kWh history
    table. Returns {top,bottom,left,right} or None. This is more reliable than the
    bar-chart plot box, which can land on a different chart (e.g. Toronto Hydro's
    lower Time-of-Use comparison) instead of the printed read-date table."""
    try:
        resp = client.messages.create(
            model=EXTRACT_MODEL,   # Sonnet localizes the table far better than Haiku
            max_tokens=200,
            messages=[{"role": "user",
                       "content": image_content + [{"type": "text", "text": HISTORY_BBOX_PROMPT}]}],
        )
        box = clean_json(resp.content[0].text)
        if not isinstance(box, dict):
            return None
        keys = ("top", "bottom", "left", "right")
        if not all(isinstance(box.get(k), (int, float)) for k in keys):
            return None
        # Guard: the history list has many rows. A box the model claims holds only a
        # row or two is the single meter-reading summary, not the history — reject it.
        rc = box.get("row_count")
        if isinstance(rc, (int, float)) and rc < 6:
            print(f"[history-consensus] locate rejected: row_count={rc} (too few rows)")
            return None
        return {k: float(box[k]) for k in keys}
    except Exception as e:
        print(f"[history-consensus] table-locate failed: {e}")
        return None


def crop_history_region(pil_image, box, upscale=2.5, save_path=None):
    """Crop the usage-history table out of the full-res original page and zoom in,
    so the small printed kWh column resolves clearly for OCR (and there are no bars
    left in frame to estimate from). `box` is {top,bottom,left,right} page fractions.
    Returns a base64 PNG, or None on failure."""
    try:
        if not box:
            return None
        W, H = pil_image.size
        l, t = box.get("left"), box.get("top")
        r, b = box.get("right"), box.get("bottom")
        if None in (l, t, r, b):
            return None
        # Light padding so nothing on the edges gets clipped.
        l = max(0.0, l - 0.03); t = max(0.0, t - 0.03)
        r = min(1.0, r + 0.03); b = min(1.0, b + 0.03)
        if r - l < 0.05 or b - t < 0.03:
            return None
        box_px = (int(l * W), int(t * H), int(r * W), int(b * H))
        crop = pil_image.crop(box_px).convert("RGB")
        if upscale and upscale > 1:
            crop = crop.resize((int(crop.width * upscale), int(crop.height * upscale)),
                               Image.LANCZOS)
        if save_path:
            try:
                crop.save(save_path)
                print(f"[history-consensus] saved crop for inspection: {save_path} "
                      f"(box pct L{l:.2f} T{t:.2f} R{r:.2f} B{b:.2f})")
            except Exception:
                pass
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[history-consensus] crop failed: {e}")
        return None


def reconcile_printed_history(client, image_content, base_history, total_kwh=None,
                              focus_b64=None):
    """Re-read the PRINTED usage-history kWh values from a zoomed crop of the
    number column and use them over the base extraction.

    Printed numbers are ground truth, but the base extraction reads the whole page
    and can digit-slip. One focused re-read against the zoomed number column (no
    bars to estimate from) is far more legible, so we take that read's value per
    row and fall back to the base value only for rows it misses. The newest row is
    then anchored to the bill's printed current-period total.

    base_history: [{date, kwh}] from the main extraction pass (fallback source).
    Returns a corrected [{date, kwh}] in base_history's order. On any failure it
    returns base_history unchanged, so this can only improve accuracy.
    """
    if not base_history:
        return base_history

    print(f"[history-consensus] running on {len(base_history)} rows"
          + (" (zoomed crop)" if focus_b64 else " (full page)"))

    # Re-read against the zoomed crop when available — the digits are far more
    # legible there — otherwise fall back to the full-page image.
    crop_content = ([{"type": "image",
                      "source": {"type": "base64", "media_type": "image/png",
                                 "data": focus_b64}}] if focus_b64 else None)

    def _read_once(temperature, content):
        try:
            resp = client.messages.create(
                model=EXTRACT_MODEL,
                max_tokens=1024,
                temperature=temperature,   # >0 decorrelates digit slips between reads
                messages=[{"role": "user",
                           "content": content + [{"type": "text", "text": HISTORY_PROMPT}]}],
            )
            parsed = clean_json(resp.content[0].text)
            if isinstance(parsed, dict):
                parsed = (parsed.get("monthly_usage_history")
                          or parsed.get("history") or [])
            rows = parsed if isinstance(parsed, list) else []
            print(f"[history-consensus]   read returned {len(rows)} rows")
            return rows
        except Exception as e:
            print(f"[history-consensus]   read failed: {e}")
            return []

    # The crop should contain most of the history rows. If a crop read returns far
    # fewer (e.g. it landed on the single meter-reading row), it's not the history
    # table — fall back to the full page rather than trusting a 1-row crop.
    min_crop_rows = max(5, len(base_history) // 2)

    def _reread(temperature):
        if crop_content is not None:
            rows = _read_once(temperature, crop_content)
            if len(rows) >= min_crop_rows:
                return rows
            print(f"[history-consensus]   crop gave {len(rows)} rows "
                  f"(<{min_crop_rows}) -> retrying on full page")
        return _read_once(temperature, image_content)

    def _tally(rows, votes):
        for r in rows:
            if not isinstance(r, dict):
                continue
            k = _hist_key(r.get("date", ""))
            v = r.get("kwh")
            if k is None or not isinstance(v, (int, float)):
                continue
            votes.setdefault(k, []).append(int(round(v)))

    # The zoomed re-read looks at the clean number column (no bars to estimate
    # from), so it's the most reliable source for the printed digits. A single
    # deterministic read is enough for Toronto's clearly-printed list; the base
    # extraction is only a fallback for rows the re-read misses, and the newest row
    # is anchored to the bill's printed total below.
    rr_votes = {}
    _tally(_reread(0.0), rr_votes)

    # Show the ballot so disagreements are visible in the logs.
    for r in base_history:
        k = _hist_key(r.get("date", ""))
        if k and k in rr_votes and len(set(rr_votes[k])) > 1:
            print(f"[history-consensus]   split {r.get('date')}: re-read votes {rr_votes[k]}")

    out, changed = [], 0
    for r in base_history:
        k = _hist_key(r.get("date", ""))
        cur = r.get("kwh")
        # Prefer the majority of the focused re-reads; fall back to the base value.
        if k is not None and k in rr_votes:
            winner = Counter(rr_votes[k]).most_common(1)[0][0]
        else:
            out.append(r)
            continue
        if isinstance(cur, (int, float)) and winner != int(round(cur)):
            changed += 1
            print(f"[history-consensus] {r.get('date')}: {int(round(cur))} -> {winner} "
                  f"(re-read votes {rr_votes[k]})")
        out.append({**r, "kwh": winner})

    # Deterministic anchor: the newest row equals the bill's current-period total,
    # which is read from the charges section — a different, more reliable region of
    # the bill than the chart. Trust it over any chart transcription.
    if isinstance(total_kwh, (int, float)) and out:
        keyed = [(_hist_key(r.get("date", "")), r) for r in out]
        keyed = [(k, r) for k, r in keyed if k is not None]
        if keyed:
            _, newest = max(keyed, key=lambda kr: kr[0])
            if newest.get("kwh") != int(round(total_kwh)):
                print(f"[history-consensus] anchor newest {newest.get('date')}: "
                      f"{newest.get('kwh')} -> {int(round(total_kwh))} (bill total)")
                newest["kwh"] = int(round(total_kwh))

    if changed:
        print(f"[history-consensus] corrected {changed} row(s) by majority vote")
    return out


def extract_with_claude(images_b64, pil_images=None):
    """
    Extract bill data from a Toronto Hydro bill's images using Claude.
    The usage-history kWh values are printed as text, so when pil_images are
    provided the printed history is re-transcribed and majority-voted for accuracy.
    """
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

    # --- Chart meta pass + main extraction (run concurrently) ---
    # The meta pass (detects that the usage history prints its kWh as text, plus the
    # issuer / current-period total) and the main field extraction are independent,
    # so fire them in parallel to cut wall-clock time. Toronto Hydro bills print each
    # period's kWh next to a "Read Date" list, so the meta pass just tells us to use
    # the printed-history consensus path.
    def _meta_pass():
        resp = client.messages.create(
            model=META_MODEL,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": image_content + [{"type": "text", "text": CHART_META_PROMPT}]
            }]
        )
        return clean_json(resp.content[0].text)

    def _extract_pass():
        resp = client.messages.create(
            model=EXTRACT_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": image_content + [{"type": "text", "text": EXTRACT_PROMPT}]
            }]
        )
        return clean_json(resp.content[0].text)

    chart_meta = {}           # chart metadata (issuer, page, printed flag)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_meta = ex.submit(_meta_pass) if pil_images else None
        fut_data = ex.submit(_extract_pass)
        if fut_meta is not None:
            try:
                chart_meta = fut_meta.result() or {}
            except Exception as e:
                print(f"[chart meta pass] failed: {e}")
                chart_meta = {}
        data = fut_data.result()

    _has_printed = bool(chart_meta.get("has_printed_numbers"))
    if pil_images:
        print(f"[chart meta] has_bar_chart={chart_meta.get('has_bar_chart')} "
              f"has_printed_numbers={chart_meta.get('has_printed_numbers')} "
              f"page_index={chart_meta.get('page_index')} "
              f"issuer={chart_meta.get('issuer')!r} "
              f"current_period_kwh={chart_meta.get('current_period_kwh')}")

    # Printed-number history: the kWh values are printed as text (ground truth), but
    # a single read occasionally digit-slips. Re-transcribe and majority-vote the
    # history so those slips can't reach the table.
    if _has_printed and (data.get("monthly_usage_history") or []):
        focus_b64 = None
        if pil_images:
            pidx = min(chart_meta.get("page_index", 0) or 0, len(pil_images) - 1)
            # Prefer the pinned per-issuer box (Toronto Hydro's list sits top-right)
            # — no API call. Only ask the model to locate the table when the issuer
            # is unknown, so the common Toronto path saves a round-trip.
            issuer = (chart_meta.get("issuer") or "").lower()
            box = None
            for name, b in HISTORY_TABLE_BOXES.items():
                if name in issuer:
                    box = b
                    print(f"[history-consensus] using pinned box for '{name}'")
                    break
            if not box:
                box = locate_history_table(client, image_content)
            print(f"[history-consensus] table box: {box}")
            focus_b64 = crop_history_region(
                pil_images[pidx], box,
                save_path=os.path.join(os.path.dirname(__file__), "_history_crop.png"),
            )
        data["monthly_usage_history"] = reconcile_printed_history(
            client, image_content,
            data.get("monthly_usage_history") or [],
            total_kwh=data.get("total_kwh"),
            focus_b64=focus_b64,
        )

    # --- Pass 2: verification (conditional) ---
    # A second model re-read mainly catches digit slips in the charges. When pass 1
    # is already internally consistent — a total is known AND the TOU/tier
    # components sum to it — that's a strong signal it's right, so skip the extra
    # call to save time. Re-verify only when no total is available or the parts
    # don't reconcile.
    _bt = (data.get("bill_type") or "").upper()
    if "TIER" in _bt:
        _parts = [data.get(k) for k in ("tier1_kwh", "tier2_kwh")]
    else:
        _parts = [data.get(k) for k in
                  ("on_peak_kwh", "mid_peak_kwh", "off_peak_kwh", "overnight_kwh")]
    _parts = [p for p in _parts if isinstance(p, (int, float))]

    # The meta pass independently reports current_period_kwh (the meter "kWh used"
    # or the summed TOU/tier components). If the extraction didn't capture total_kwh
    # but the components sum to that independent figure, the two agree — no need to
    # burn a verify call. Backfill total_kwh from it so the output table is complete.
    _total = data.get("total_kwh")
    if not _total and chart_meta.get("current_period_kwh"):
        _total = data["total_kwh"] = chart_meta["current_period_kwh"]
    if not _total and _parts:
        _total = data["total_kwh"] = round(sum(_parts))

    if not _total or not _parts:
        needs_verify = True
    else:
        needs_verify = abs(sum(_parts) - _total) > 0.02 * max(_total, 1)

    if needs_verify:
        verify_prompt = VERIFY_PROMPT.format(data=json.dumps(data, indent=2))
        response2 = client.messages.create(
            model=VERIFY_MODEL,
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": image_content + [{"type": "text", "text": verify_prompt}]
            }]
        )
        raw2 = response2.content[0].text
        print(f"[verify] ran (pass-1 inconsistent); response ({len(raw2)} chars): {raw2[:200]}")
        verified = clean_json(raw2)
    else:
        print(f"[verify] skipped (pass-1 consistent: components sum to total {_total})")
        verified = dict(data)

    return verified


def parse_history_date(date_str):
    """Parse a wide range of history-bar date labels → (year, month_index 0-11).
    Handles 'DD MMM YY', 'MMM YY', 'MMM YYYY', 'April 2025', '04/2025',
    '2025-04', '26-Feb-26', etc."""
    if not date_str:
        return None, None
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", str(date_str).strip(), flags=re.I)
    s = re.sub(r"\s+", " ", s).replace(",", "")
    fmts = (
        "%d %b %y", "%d %b %Y", "%d-%b-%y", "%d-%b-%Y",
        "%b %y", "%b %Y", "%B %y", "%B %Y",
        "%b-%y", "%b-%Y", "%b/%y", "%b/%Y",
        "%m/%Y", "%m-%Y", "%Y-%m", "%m/%d/%Y", "%m/%d/%y",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.year, dt.month - 1
        except ValueError:
            continue
    # Fallback for messy / billing-period range labels like "Mar 19-25" or
    # "Feb 19-26" (Elexicon-style "<month> <day>-<yy>"): take the month from the
    # first alphabetic token and the year from the LAST numeric token (the year
    # follows the day range — e.g. Jan 19-26 -> 2026, Dec 19-25 -> 2025).
    nums = re.findall(r"\d+", s)
    alpha = re.search(r"[A-Za-z]{3,}", s)
    if alpha and nums:
        abbr3 = [m.lower() for m in MONTH_ABBR]
        key = alpha.group(0).lower()[:3]
        if key in abbr3:
            yr = int(nums[-1])
            if yr < 100:
                yr += 2000
            if 2000 <= yr <= 2100:
                return yr, abbr3.index(key)
    return None, None


def build_monthly_history(raw_history, billing_period_end=None):
    """
    Takes list of {date, kwh}, keeps most recent 12 months in chronological order.
    Fills gaps (months with no data in the chart) with kwh=null so the table
    always shows a complete month range.
    Each output row: {month_abbr, month_index, kwh}

    billing_period_end: 'MM/DD/YYYY' string — if provided, anchors the 12-month
    window to this date so charts with 13 bars (e.g. Mar 25 → Mar 26) don't
    accidentally include the oldest bar (Mar 25) when the newest bar (Mar 26)
    is missing from pixel extraction.
    """
    if not raw_history:
        return []

    parsed = []
    for entry in raw_history:
        date_str = entry.get("date", "")
        year, month_idx = parse_history_date(date_str)
        if year is not None:
            # Day-of-month, used only to order two reads that fall in the same
            # calendar month (e.g. 07 Oct vs 31 Oct) so the earlier read keeps the
            # month and the later one rolls forward. Day-less labels ("Mar 25")
            # sort mid-month, which is fine since they never collide.
            dm = re.match(r"\s*(\d{1,2})[\s\-/](?=[A-Za-z])", str(date_str))
            day = int(dm.group(1)) if dm else 15
            parsed.append({
                "year": year,
                "month_index": month_idx,
                "month_abbr": MONTH_ABBR[month_idx],
                "day": day,
                "kwh": entry.get("kwh"),
                "days_fallback": entry.get("days_fallback"),
            })

    if not parsed:
        return []

    parsed.sort(key=lambda x: (x["year"], x["month_index"], x["day"]))

    # Meter-read history charts (e.g. Toronto Hydro's "Read Date" usage list) label
    # each bar by the date the meter was read, and that date drifts within a month.
    # Two reads can land in the same calendar month — e.g. 07 Oct and 31 Oct — which
    # would collapse into one bucket and leave the following month (Nov) blank. Each
    # read is one ~monthly billing period, so enforce strictly-increasing month
    # buckets: when an entry's month is already taken by an earlier read, advance it
    # to the next month. Charts that already have exactly one bar per month are
    # strictly increasing, so this is a no-op for them.
    def _next_month(y, m):
        m += 1
        if m >= 12:
            m, y = 0, y + 1
        return y, m

    last_ym = None
    for e in parsed:
        ym = (e["year"], e["month_index"])
        if last_ym is not None and ym <= last_ym:
            ym = _next_month(*last_ym)
        e["year"], e["month_index"] = ym
        e["month_abbr"] = MONTH_ABBR[ym[1]]
        last_ym = ym

    # Determine window endpoint: prefer billing_period_end over last data point.
    anchor_y, anchor_m = None, None
    if billing_period_end:
        try:
            dt = datetime.strptime(billing_period_end.strip(), "%m/%d/%Y")
            anchor_y, anchor_m = dt.year, dt.month - 1
        except Exception:
            pass

    if anchor_y is not None:
        latest_y = parsed[-1]["year"]
        latest_m = parsed[-1]["month_index"]
        # Anchor the 12-month window to the latest ACTUAL data point (the newest
        # bar). The dedicated extractors reliably capture every bar, so the newest
        # bar is the true most-recent month — even when billing_period_end is
        # labelled a month later than the chart (e.g. Enova labels the Sep 11–Oct 11
        # period as "Sep", so the chart ends Sep 24 while the bill ends Oct 2024).
        # Anchoring to billing_period_end there invents an empty trailing month and
        # pushes a real month out of the window.
        end_y, end_m = latest_y, latest_m
        # Keep only entries within the 12-month window ending at end_y/end_m.
        start_m, start_y = end_m, end_y
        for _ in range(11):
            start_m -= 1
            if start_m < 0:
                start_m = 11
                start_y -= 1
        recent = [e for e in parsed
                  if (start_y, start_m) <= (e["year"], e["month_index"]) <= (end_y, end_m)]
    else:
        recent = parsed[-12:]
        end_y, end_m = recent[-1]["year"], recent[-1]["month_index"]

    # Fill a full 12-month window ending at the last data point.
    # This ensures months with no chart data (e.g. Aug/Sep in a rolling history)
    # still appear as null rows.
    existing = {(e["year"], e["month_index"]): e for e in recent}
    # Walk back 11 months to find window start
    start_m, start_y = end_m, end_y
    for _ in range(11):
        start_m -= 1
        if start_m < 0:
            start_m = 11
            start_y -= 1

    filled = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        filled.append(existing.get((y, m), {
            "year": y,
            "month_index": m,
            "month_abbr": MONTH_ABBR[m],
            "kwh": None
        }))
        m += 1
        if m >= 12:
            m = 0
            y += 1

    filled.sort(key=lambda x: x["month_index"])
    return filled


def month_from_date(date_str):
    """Parse 'MM/DD/YYYY' and return (month_name, year)."""
    if not date_str:
        return None, None
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")
        return dt.strftime("%B"), dt.year
    except Exception:
        return None, None


def fmt_date_display(date_str):
    """Convert 'MM/DD/YYYY' -> 'Jun 29, 2026' for display.
    Returns '' for empty input and passes the original string through unchanged
    if it can't be parsed."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(str(date_str).strip(), "%m/%d/%Y")
        return dt.strftime("%b %d, %Y")
    except Exception:
        return date_str


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

    # All files in a single upload belong to ONE customer record. Some utilities
    # (e.g. Tillsonburg) put the usage-history chart in a SEPARATE file from the
    # charges, so we treat every uploaded file (in practice one or two — bill +
    # chart) as pages of one bill: charges are read from whichever page has them,
    # the history chart from whichever page has it.
    valid_files = [f for f in files
                   if f.filename.lower().rsplit(".", 1)[-1] in ("pdf", "png")]
    groups = {"__record__": valid_files} if valid_files else {}
    group_order = ["__record__"] if valid_files else []

    bills = []

    for key in group_order:
        group_files = groups[key]
        all_pil, all_b64, tmp_paths = [], [], []

        try:
            for f in group_files:
                tmp_path = f"/tmp/{uuid.uuid4().hex}_{f.filename}"
                f.save(tmp_path)
                tmp_paths.append(tmp_path)
                pil_images, images_b64 = file_to_images(tmp_path)
                all_pil.extend(pil_images)
                all_b64.extend(images_b64)

            data = extract_with_claude(all_b64, all_pil)
        except ValueError as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            import traceback; traceback.print_exc()
            label = group_files[0].filename
            return jsonify({"error": f"Failed to process {label}: {str(e)}"}), 500
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)

        # Billing month: prefer the period end date; fall back to the bill /
        # statement date when no period range is printed (e.g. Halton, Enova).
        month, year = month_from_date(data.get("billing_period_end"))
        if not month:
            month, year = month_from_date(data.get("bill_date"))
        monthly_history = build_monthly_history(
            data.get("monthly_usage_history") or [],
            billing_period_end=data.get("billing_period_end")
        )

        if len(group_files) == 1:
            display_name = group_files[0].filename
        else:
            display_name = f"{group_files[0].filename} (+{len(group_files)-1} page{'s' if len(group_files)>2 else ''})"

        bills.append({
            "filename": display_name,
            "bill_type": data.get("bill_type", "TOU"),
            "billing_month": month or "Unknown",
            "billing_year": year or 0,
            "period_start": fmt_date_display(data.get("billing_period_start")),
            "period_end": fmt_date_display(data.get("billing_period_end")),
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

        # Release this bill's large image arrays before processing the next one,
        # so peak memory stays flat instead of accumulating across bills.
        del all_pil, all_b64
        gc.collect()

    if not bills:
        return jsonify({"error": "No valid PDF or PNG files found"}), 400

    rows = build_sorted_rows(bills)
    return jsonify({"rows": rows})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

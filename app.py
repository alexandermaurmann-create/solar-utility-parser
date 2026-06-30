import os
import io
import re
import gc
import json
import base64
import uuid
from itertools import combinations
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
  "history_is_daily_average": true or false,
  "monthly_usage_history": [
    {"date": "DD MMM YY", "kwh": number, "days": number or null},
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
- history_is_daily_average = set TRUE only when the usage history gives ONLY daily-average values and NO monthly total is available — i.e. the y-axis / column is "kWh per day" / "KWH per day" / "Daily Average" / "Avg/Day" and there is no separate monthly-total column. Set FALSE when a monthly kWh total is available. CRITICAL: some bills print BOTH a total "kWh" column AND a per-day ("/ Day", "kWh/Day", "/Day") column for each period, side by side — in that case set this FALSE and read the TOTAL kWh column, NEVER the per-day number. Only when a per-day value is the sole number shown is this true.
- monthly_usage_history = ALL bars from the usage history chart (typically 13-15 entries). For EACH bar return:
  * "kwh" = the value for that bar EXACTLY as printed/shown — do NOT multiply, convert, or adjust it. If BOTH a total-kWh column and a per-day column are printed, use the TOTAL-kWh value here (e.g. read 1401, not 43.78). Only when the bar shows just a daily-average number (no total) return that daily number, e.g. 46, and we will multiply by days ourselves.
  * "days" = that bar's number of billing days, read from the "# of Days" / "# days" row under the chart. REQUIRED when history_is_daily_average is true; use null otherwise.
  * "date" = each bar's date, formatted "DD MMM YY" (e.g. "26 FEB 26"). ALWAYS include a year. If the chart labels bars with only a month name (e.g. "Apr", "May"), infer each bar's year from the chart's date range or title (e.g. a title "APRIL 2025 - APRIL 2026" means the bars run Apr 2025 → Apr 2026) and output e.g. "01 APR 25". Never return a bare month with no year.
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
  "issuer": "utility company name printed on the bill, lowercase (e.g. elexicon, burlington hydro, toronto hydro), or null",
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
- issuer: the electricity utility that issued the bill, read from the logo/header (e.g. "elexicon", "burlington hydro", "toronto hydro", "tillsonburg hydro"). Lowercase. null if unclear. (Some bills omit the brand name — that's fine, return null and rely on hst_number.)
- hst_number: the HST / GST tax registration number printed on the bill, usually near "HST" or "Reg. no." (e.g. "86360 3726 RT0001"). This identifies the utility even when the brand name is absent. null if not found.

If there is no bar chart, return {"has_bar_chart": false}.
"""


# Per-utility overrides that PIN the brittle, non-deterministic parts of chart
# pixel extraction.  The pixel math is reliable given a correct chart crop, but
# Claude's per-run crop wobbles (Opus 4.8 has no temperature control), which
# occasionally truncates the chart and corrupts extraction.  For known utilities
# we pin the crop region and bar count so extraction is identical every run.
# Gridline VALUES still come from Claude — it reads y-axis labels reliably; only
# the geometry is pinned.  Keyed by a substring of the issuer Claude reads.
# Crop = (left_pct, top_pct, right_pct, bottom_pct).  Calibrated per bill template.
CHART_PROFILES = {
    "elexicon": {
        "chart_crop_pct": (0.20, 0.79, 0.97, 0.965),
        "bar_count": 13,
    },
    "tillsonburg": {
        # 3-colour stacked chart (On/Mid/Off-peak), separate image, no printed
        # numbers, faint gridlines → stacked extractor (calibrate-by-total).
        "extractor": "stacked",
    },
    "milton": {
        # Mixed Tiered (solid green) + TOU (3-colour stacked) monthly bars,
        # separate image, no printed numbers → same stacked extractor.
        "extractor": "stacked",
    },
    "guelph": {
        # Single-colour "kWh per day" (daily-average) chart, separate image,
        # crisp gridlines. Reusable daily-average extractor: reads daily avgs
        # off the gridlines and ×days-in-month → monthly kWh.
        "extractor": "dailyavg",
    },
    "halton": {
        # Same daily-average archetype as Guelph, but GRAY bars/gridlines.
        "extractor": "dailyavg",
    },
    "enova": {
        # Dual-bar "$ vs kWh" chart, no printed numbers, and the bill shows NO
        # brand name — so match on the HST number instead of the issuer.
        "extractor": "dualbar",
        "hst": "863603726",
    },
}


def apply_chart_profile(chart_meta):
    """If the issuer matches a known profile, pin its crop + bar_count so the
    pixel extraction stops depending on Claude's non-deterministic geometry."""
    issuer = (chart_meta.get("issuer") or "").lower()
    hst = re.sub(r"\D", "", chart_meta.get("hst_number") or "")[:9]
    for key, prof in CHART_PROFILES.items():
        prof_hst = re.sub(r"\D", "", prof.get("hst", ""))[:9]
        if (key in issuer) or (prof_hst and hst and prof_hst == hst):
            if "chart_crop_pct" in prof:
                l, t, r, b = prof["chart_crop_pct"]
                chart_meta["chart_left_pct"]  = l
                chart_meta["chart_top_pct"]   = t
                chart_meta["chart_right_pct"] = r
                chart_meta["chart_bottom_pct"] = b
            if "bar_count" in prof:
                chart_meta["bar_count"] = prof["bar_count"]
            if "extractor" in prof:
                chart_meta["extractor"] = prof["extractor"]
            print(f"[profile] applied '{key}'"
                  + (f" → extractor={prof['extractor']}" if "extractor" in prof
                     else ": pinned crop + bar_count"))
            return chart_meta
    return chart_meta


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


def pixel_extract_bars(pil_image, meta):
    """
    Extract bar heights using pixel analysis.
    Improvements:
      1. 4x zoom for finer sub-pixel precision
      2. Weighted gridline centroid (coverage-weighted, not simple mean)
      3. K-means auto color detection (median of colorful pixels in plot area)
      4. Center 60% column sampling (avoids anti-aliased bar edges)
      5. Gradient-based bar top refinement
      6. Outlier re-examination (re-test bars >2x neighbors at higher threshold)
      7. Total kWh cross-validation and scaling correction

    Returns list of {"date": "01 MMM YY", "kwh": int} or None if analysis fails.
    """
    try:
        import numpy as np

        orig_w, orig_h = pil_image.size

        # Crop to Claude's estimated chart area
        left   = max(0, int(meta["chart_left_pct"]   * orig_w))
        right  = min(orig_w, int(meta["chart_right_pct"] * orig_w))
        top    = max(0, int(meta["chart_top_pct"]    * orig_h))
        bottom = min(orig_h, int(meta["chart_bottom_pct"] * orig_h))

        chart_orig_pil = pil_image.crop((left, top, right, bottom)).convert("RGB")
        orig_ch = chart_orig_pil.height
        orig_cw = chart_orig_pil.width

        # --- 1. ZOOM 3x ---
        SCALE = 3
        abs_off = top * SCALE   # zoomed pixels from image top to crop top (constant)
        chart_pil = chart_orig_pil.resize(
            (orig_cw * SCALE, orig_ch * SCALE), Image.LANCZOS
        )
        chart = np.array(chart_pil)
        ch, cw = chart.shape[:2]

        if ch < 30 or cw < 30:
            return None

        y_min        = float(meta.get("y_axis_min", 0))
        y_max        = float(meta.get("y_axis_max", 5000))
        month_labels = meta.get("month_labels", [])
        bar_count    = meta.get("bar_count")
        gl_values    = meta.get("y_axis_gridlines") or []
        bar_color    = meta.get("bar_color_rgb")
        chart_total  = meta.get("chart_total_kwh")

        # If Claude counted fewer bars than labels, trim oldest labels to match.
        if bar_count and isinstance(bar_count, int) and 0 < bar_count < len(month_labels):
            print(f"[pixel] trimming month_labels from {len(month_labels)} to bar_count={bar_count}")
            month_labels = month_labels[-bar_count:]

        num_bars = len(month_labels)
        if num_bars == 0 or y_max <= y_min:
            return None

        r = chart[:, :, 0].astype(float)
        g = chart[:, :, 1].astype(float)
        b = chart[:, :, 2].astype(float)
        brightness = r + g + b

        # --- 2. WEIGHTED GRIDLINE CENTROID ---
        # Cluster rows into gridline candidates using coverage-weighted centroid
        # instead of a simple mean — gives sub-pixel accurate gridline position.
        def cluster_rows(rows, gap, weights=None):
            if len(rows) == 0:
                return []
            clusters, cur = [], [int(rows[0])]
            cur_w = [float(weights[rows[0]])] if weights is not None else []
            for row in rows[1:]:
                if row - cur[-1] <= gap:
                    cur.append(int(row))
                    if weights is not None:
                        cur_w.append(float(weights[row]))
                else:
                    if weights is not None and sum(cur_w) > 0:
                        clusters.append(float(np.average(cur, weights=cur_w)))
                    else:
                        clusters.append(float(np.mean(cur)))
                    cur = [int(row)]
                    cur_w = [float(weights[row])] if weights is not None else []
            if weights is not None and sum(cur_w) > 0:
                clusters.append(float(np.average(cur, weights=cur_w)))
            else:
                clusters.append(float(np.mean(cur)))
            return clusters

        # --- GRIDLINE-BASED Y CALIBRATION ---
        # Detect gridlines in the FULL bill image (restricted to chart columns),
        # with a generous row window that extends 5px above and 30px below Claude's
        # crop.  Using absolute pixel coordinates means the calibration is stable
        # regardless of how much chart_*_pct drifts between Claude runs — the
        # gridlines are always at the same absolute positions in the original image.
        bar_centers_pct = meta.get("bar_centers_pct") or []
        gl_candidates   = []
        row_to_kwh      = None
        all_gray_cands  = []

        if gl_values and len(gl_values) >= 2:
            orig_arr = np.array(chart_orig_pil)
            ro = orig_arr[:, :, 0].astype(float)
            go = orig_arr[:, :, 1].astype(float)
            bo = orig_arr[:, :, 2].astype(float)
            max_diff_o = np.maximum(np.maximum(np.abs(ro-go), np.abs(go-bo)), np.abs(ro-bo))
            avg_o = (ro + go + bo) / 3
            is_gray_o = (max_diff_o < 30) & (avg_o > 60) & (avg_o < 240)
            gray_cov_o = is_gray_o.mean(axis=1)

            print(f"[pixel] gridline coverage: max={gray_cov_o.max():.3f} mean={gray_cov_o.mean():.3f}")

            # Use absolute image coordinates for calibration so the slope is
            # crop-offset-independent: gl_px uses (crop_row + top)*SCALE and
            # row_to_kwh adds top*SCALE to convert the crop-zoomed bar_top to absolute.
            abs_off = top * SCALE  # zoomed pixels from image top to crop top

            for gl_thresh in [0.50, 0.40, 0.30, 0.20, 0.12]:
                cands = cluster_rows(np.where(gray_cov_o > gl_thresh)[0], gap=3, weights=gray_cov_o)
                print(f"[pixel] gridline thresh={gl_thresh}: {len(cands)} candidates "
                      f"(need {len(gl_values)}): rows={[round(c,1) for c in cands]}")
                if not all_gray_cands and len(cands) >= 2:
                    all_gray_cands = cands
                if len(cands) == len(gl_values):
                    gl_candidates = cands
                    gl_px   = np.array([(row + top) * SCALE for row in gl_candidates], dtype=float)
                    gl_vals = np.array(gl_values, dtype=float)
                    coeffs  = np.polyfit(gl_px, gl_vals, 1)
                    row_to_kwh = lambda row, c=coeffs, off=abs_off: float(np.polyval(c, row + off))
                    print(f"[pixel] gridline calibration OK slope={coeffs[0]:.4f} intercept={coeffs[1]:.1f}")
                    break

            # all_gray_cands is in CROP-relative coords; crop_to_slice = 0 here
            # (keeping variable for gap-fill compat)
            crop_to_slice = 0

            # --- INFER MISSING GRIDLINES IN LARGE GAPS ---
            # Tall bars can cover a gridline, making it invisible at normal thresholds.
            # If any gap between consecutive candidates is >1.5x the median gap,
            # scan within that gap at very low threshold to recover the hidden line.
            if row_to_kwh is None and len(all_gray_cands) >= 2:
                sorted_ac = sorted(all_gray_cands)
                gaps = [sorted_ac[i+1] - sorted_ac[i] for i in range(len(sorted_ac)-1)]
                median_gap = float(np.median(gaps))
                extra_found = False
                for i, gap in enumerate(gaps):
                    if gap > median_gap * 1.5 and median_gap > 3:
                        # A bar may fully cover this gridline, making coverage ≈0.
                        # Don't rely on a coverage peak — just place the missing row
                        # at the midpoint of the gap (gridlines are evenly spaced).
                        inferred = (sorted_ac[i] + sorted_ac[i+1]) / 2.0
                        all_gray_cands.append(inferred)
                        extra_found = True
                        print(f"[pixel] inferred hidden gridline at row {inferred:.1f} "
                              f"(midpoint of gap {sorted_ac[i]:.1f}–{sorted_ac[i+1]:.1f})")
                # Re-try exact match after adding inferred row
                if extra_found:
                    new_cands = cluster_rows(
                        np.array(sorted(int(c) for c in all_gray_cands)), gap=3, weights=gray_cov_o
                    )
                    if len(new_cands) == len(gl_values):
                        gl_candidates = new_cands
                        gl_px   = np.array([(row + top) * SCALE for row in gl_candidates], dtype=float)
                        gl_vals = np.array(gl_values, dtype=float)
                        coeffs  = np.polyfit(gl_px, gl_vals, 1)
                        row_to_kwh = lambda row, c=coeffs, off=abs_off: float(np.polyval(c, row + off))
                        print(f"[pixel] gridline calibration OK after gap-fill: "
                              f"slope={coeffs[0]:.4f} intercept={coeffs[1]:.1f}")

                    elif len(new_cands) == len(gl_values) - 1 and len(new_cands) >= 4:
                        # Claude over-counted by 1 (commonly adds a fabricated top value
                        # like 800 when the chart only goes to 700).  Try trimming the
                        # top value, then the bottom value, and accept whichever gives a
                        # near-perfect linear fit.
                        for trimmed, trim_label in [
                            (gl_values[1:],  'top'),
                            (gl_values[:-1], 'bottom'),
                        ]:
                            gl_px  = np.array([(r + top) * SCALE for r in new_cands], dtype=float)
                            gl_v   = np.array(trimmed, dtype=float)
                            c      = np.polyfit(gl_px, gl_v, 1)
                            res    = np.abs(np.polyval(c, gl_px) - gl_v).max()
                            if res < 15:
                                gl_candidates = list(new_cands)
                                row_to_kwh = lambda row, cf=c, off=abs_off: float(np.polyval(cf, row + off))
                                print(f"[pixel] calibration OK (trimmed {trim_label} value): "
                                      f"slope={c[0]:.4f} intercept={c[1]:.1f}")
                                break

            # --- PARTIAL GRIDLINE CALIBRATION ---
            if row_to_kwh is None and len(all_gray_cands) >= 3:
                residual_thresh = max(40.0, 0.06 * (y_max - y_min))
                best_res, best_coeffs, best_cands = float('inf'), None, None
                sorted_cands = sorted(all_gray_cands)

                # Drop trailing rows whose gap is >1.6× the median inner gap.
                # These are usually chart-border artifacts, not real gridlines,
                # and they pull the calibration slope off if included.
                while len(sorted_cands) >= 4:
                    inner_gaps = np.diff(sorted_cands[:-1])  # gaps excluding the last
                    med_inner  = float(np.median(inner_gaps)) if len(inner_gaps) else 0
                    last_gap   = sorted_cands[-1] - sorted_cands[-2]
                    if med_inner > 0 and last_gap > med_inner * 1.6:
                        dropped = sorted_cands.pop()
                        print(f"[pixel] dropping trailing outlier row {dropped:.1f} "
                              f"(gap {last_gap:.1f} > 1.6× med {med_inner:.1f})")
                    else:
                        break

                max_sz = min(len(sorted_cands), len(gl_values))
                for sz in range(max_sz, 2, -1):
                    for cand_sub in combinations(sorted_cands, sz):
                        # Use absolute zoomed rows for calibration (crop-relative + top)
                        px_sub = np.array([(row + top) * SCALE for row in cand_sub], dtype=float)
                        for val_sub in combinations(gl_values, sz):
                            v = np.array(val_sub, dtype=float)
                            if v.max() == v.min():
                                continue
                            # Require evenly-spaced kWh values — real chart gridlines
                            # are always at equal intervals, so any subset that isn't
                            # evenly spaced (e.g. includes a fabricated extra value) is skipped.
                            v_sorted = np.sort(v)
                            diffs = np.diff(v_sorted)
                            if len(diffs) > 1:
                                expected = diffs.mean()
                                if expected > 0 and not np.allclose(diffs, expected, rtol=0.08):
                                    continue
                            c = np.polyfit(px_sub, v, 1)
                            res = np.abs(np.polyval(c, px_sub) - v).max()
                            if res < best_res:
                                best_res, best_coeffs, best_cands = res, c, cand_sub
                    if best_res < residual_thresh:
                        break

                if best_res < residual_thresh and best_coeffs is not None:
                    gl_candidates = list(best_cands)
                    row_to_kwh = lambda row, c=best_coeffs, off=abs_off: float(np.polyval(c, row + off))
                    print(f"[pixel] partial gridline fit: {len(best_cands)} rows, "
                          f"slope={best_coeffs[0]:.4f}, max_residual={best_res:.1f}")
                else:
                    print(f"[pixel] gridline calibration failed (best residual={best_res:.1f}) "
                          f"— falling back to floor detection")
            elif row_to_kwh is None:
                print("[pixel] gridline calibration failed — falling back to floor detection")

        # Did the gridline-label calibration succeed?  If so it spans the full
        # y-range and is more accurate than the 2-point bar-bottom calibration
        # below, so we must NOT let bar-bottom override it (that override placed
        # Elexicon's zero-line ~900 kWh too high, collapsing summer bars to 0).
        gridline_ok = row_to_kwh is not None

        # --- MASK BAR DETECTION TO CHART PLOT AREA ---
        if gl_candidates:
            plot_top = max(0, int(gl_candidates[0]  * SCALE) - SCALE * 3)
            plot_bot = min(ch, int(gl_candidates[-1] * SCALE) + SCALE * 5)
        elif all_gray_cands:
            skip_orig = max(1, int(orig_ch * 0.08))
            useful = [c for c in all_gray_cands if c > skip_orig]
            if len(useful) >= 2:
                plot_top = max(0, int(useful[0] * SCALE) + SCALE)
                plot_bot = min(ch, int(useful[-1] * SCALE) + SCALE * 5)
            else:
                plot_top = 0
                plot_bot = ch
        else:
            plot_top = 0
            plot_bot = ch

        # Clamp plot_top to the row of the highest CALIBRATED gridline.
        # Use row_to_kwh(gl_candidates[0] * SCALE) — the kWh the calibration assigns
        # to its topmost row — rather than Claude's y_axis_max, which is often inflated.
        if row_to_kwh is not None and gl_candidates:
            clamp_ceiling = row_to_kwh(gl_candidates[0] * SCALE)
            lo, hi = 0, ch
            while lo < hi:
                mid = (lo + hi) // 2
                if row_to_kwh(mid) > clamp_ceiling:
                    lo = mid + 1
                else:
                    hi = mid
            row_at_ymax = lo
            if row_at_ymax > plot_top:
                print(f"[pixel] plot_top clamped {plot_top}→{row_at_ymax} (ceiling={clamp_ceiling:.0f} kWh)")
                plot_top = row_at_ymax

        print(f"[pixel] plot area: rows {plot_top}–{plot_bot} of {ch} (zoomed px)")

        # --- 3. AUTO COLOR DETECTION (K-MEANS MEDIAN) ---
        # Find dominant bar color by taking the median RGB of colorful pixels in
        # the plot area — no reliance on Claude's color estimate.
        max_diff = np.maximum(np.maximum(np.abs(r-g), np.abs(g-b)), np.abs(r-b))
        is_bar_generic = (max_diff > 25) & (brightness < 660)

        plot_r = r[plot_top:plot_bot, :]
        plot_g = g[plot_top:plot_bot, :]
        plot_b = b[plot_top:plot_bot, :]
        colorful_mask = (max_diff[plot_top:plot_bot, :] > 30) & \
                        (brightness[plot_top:plot_bot, :] > 80) & \
                        (brightness[plot_top:plot_bot, :] < 580)
        is_bar_auto = None
        if colorful_mask.sum() > 100:
            auto_r = float(np.median(plot_r[colorful_mask]))
            auto_g = float(np.median(plot_g[colorful_mask]))
            auto_b = float(np.median(plot_b[colorful_mask]))
            auto_dist = np.sqrt((r - auto_r)**2 + (g - auto_g)**2 + (b - auto_b)**2)
            is_bar_auto = auto_dist < 75
            auto_count = int(is_bar_auto[plot_top:plot_bot, :].sum())
            print(f"[pixel] auto color: RGB({auto_r:.0f},{auto_g:.0f},{auto_b:.0f}), {auto_count} px")

        # --- 3b. TWO-COLOR (STACKED) BAR DETECTION ---
        # Stacked charts (e.g. Tier1/Tier2) draw each bar in two colors. A single-
        # colour mask then catches only one tier, leaving short bars (which may be
        # almost entirely the other tier) too sparse to register as bar columns —
        # the pipeline mis-counts bars and falls back to misaligned equal spacing.
        # Detect a distinct second dominant colour and, when the plot area is
        # genuinely two-tone, mask on the UNION of both tiers.
        is_bar_two = None
        if colorful_mask.sum() > 200:
            pix = np.stack([plot_r[colorful_mask], plot_g[colorful_mask],
                            plot_b[colorful_mask]], axis=1).astype(float)
            rng = np.random.default_rng(0)
            cen = pix[rng.choice(len(pix), 2, replace=False)].astype(float)
            for _ in range(15):
                lab = np.linalg.norm(pix[:, None, :] - cen[None], axis=2).argmin(1)
                for k in range(2):
                    if (lab == k).any():
                        cen[k] = pix[lab == k].mean(0)
            sep = float(np.linalg.norm(cen[0] - cen[1]))
            n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
            frac = min(n0, n1) / max(1, n0 + n1)
            # Two well-separated, both-substantial colours ⇒ stacked bar chart.
            if sep > 60 and frac > 0.15:
                d0 = np.sqrt((r - cen[0, 0])**2 + (g - cen[0, 1])**2 + (b - cen[0, 2])**2)
                d1 = np.sqrt((r - cen[1, 0])**2 + (g - cen[1, 1])**2 + (b - cen[1, 2])**2)
                is_bar_two = (d0 < 75) | (d1 < 75)
                two_count = int(is_bar_two[plot_top:plot_bot, :].sum())
                print(f"[pixel] two-color stacked chart: "
                      f"RGB({cen[0,0]:.0f},{cen[0,1]:.0f},{cen[0,2]:.0f})+"
                      f"RGB({cen[1,0]:.0f},{cen[1,1]:.0f},{cen[1,2]:.0f}), {two_count} px")

        # --- 4. BAR COLOR SELECTION ---
        # Priority: Claude fingerprint > auto-detected > generic
        # Prefer whichever is most selective while still covering enough pixels.
        gen_count = int(is_bar_generic[plot_top:plot_bot, :].sum())

        if is_bar_two is not None:
            # Stacked two-tone chart: union of both tiers is the only mask that
            # captures every bar (short bars may be entirely the second colour).
            is_bar = is_bar_two
            print("[pixel] using two-color union mask")
        elif bar_color and len(bar_color) == 3:
            bc = np.array(bar_color, dtype=float)
            color_dist = np.sqrt((r - bc[0])**2 + (g - bc[1])**2 + (b - bc[2])**2)
            is_bar_fp = color_dist < 80
            fp_count = int(is_bar_fp[plot_top:plot_bot, :].sum())
            if fp_count >= gen_count * 0.3:
                is_bar = is_bar_fp
                print(f"[pixel] using Claude fingerprint: {fp_count} px")
            elif is_bar_auto is not None:
                auto_count = int(is_bar_auto[plot_top:plot_bot, :].sum())
                if auto_count >= gen_count * 0.3:
                    is_bar = is_bar_auto
                    print(f"[pixel] fingerprint sparse → auto color: {auto_count} px")
                else:
                    is_bar = is_bar_generic
                    print(f"[pixel] falling back to generic: {gen_count} px")
            else:
                is_bar = is_bar_generic
                print(f"[pixel] fingerprint sparse → generic: {gen_count} px")
        elif is_bar_auto is not None:
            auto_count = int(is_bar_auto[plot_top:plot_bot, :].sum())
            if auto_count >= gen_count * 0.3:
                is_bar = is_bar_auto
                print(f"[pixel] using auto color (no fingerprint): {auto_count} px")
            else:
                is_bar = is_bar_generic
                print(f"[pixel] auto insufficient → generic: {gen_count} px")
        else:
            is_bar = is_bar_generic
            print(f"[pixel] generic color detection: {gen_count} px")

        # Apply plot area mask
        is_bar[:plot_top, :] = False
        is_bar[plot_bot:,  :] = False

        # --- FALLBACK: X-AXIS LINE + BAR FLOOR DETECTION ---
        if row_to_kwh is None:
            is_dark = brightness < 450
            dark_cov = is_dark.mean(axis=1)
            dark_rows = np.where(dark_cov > 0.30)[0]
            below = dark_rows[dark_rows >= plot_bot - SCALE * 8] if len(dark_rows) > 0 else []
            chart_floor = int(below.min()) if len(below) > 0 else plot_bot
            eff_height = max(1, chart_floor - plot_top)
            row_to_kwh = lambda row, t=float(plot_top), h=float(eff_height), \
                ymn=y_min, ymx=y_max: ymx - ((row - t) / h) * (ymx - ymn)
            print(f"[pixel] floor calibration: plot_top={plot_top}, floor={chart_floor}, height={eff_height}px")

        COV_THRESH = 0.3

        # --- BAR X-POSITION DETECTION ---
        half_bar_w = max(3, cw // (num_bars * 3))
        col_density = is_bar[plot_top:plot_bot, :].sum(axis=0).astype(float) / max(plot_bot - plot_top, 1)
        min_density = max(0.03, col_density.max() * 0.10)
        in_bar_col  = col_density > min_density
        bar_cols    = np.where(in_bar_col)[0]

        bar_ranges = []
        if len(bar_cols) >= 3:
            bar_start = int(bar_cols[0])
            bar_end   = int(bar_cols[-1])

            diffs      = np.diff(in_bar_col.astype(int))
            grp_starts = list(np.where(diffs == 1)[0] + 1)
            grp_ends   = list(np.where(diffs == -1)[0] + 1)
            if in_bar_col[0]:
                grp_starts.insert(0, 0)
            if in_bar_col[-1]:
                grp_ends.append(bar_end + 1)

            min_grp_w  = max(4, (bar_end - bar_start) // (num_bars * 2))
            bar_groups = [(s, e) for s, e in zip(grp_starts, grp_ends) if e - s >= min_grp_w]
            pixel_bar_count = len(bar_groups)

            print(f"[pixel] bar region x={bar_start}–{bar_end}, "
                  f"pixel groups={pixel_bar_count}, Claude said {num_bars}")

            if max(3, num_bars // 2) <= pixel_bar_count <= num_bars + 2:
                bar_centers = [(gs + ge) // 2 for gs, ge in bar_groups]

                if pixel_bar_count != num_bars:
                    # If exactly one bar is missing, find the double-sized gap and
                    # insert an inferred center there — don't trim the oldest label.
                    missing_inserted = False
                    if pixel_bar_count == num_bars - 1 and len(bar_centers) >= 2:
                        steps = [bar_centers[i+1] - bar_centers[i]
                                 for i in range(len(bar_centers)-1)]
                        med_step = float(np.median(steps))
                        for gap_i, step in enumerate(steps):
                            if step > med_step * 1.5:
                                inferred_cx = int(bar_centers[gap_i] + med_step)
                                bar_centers.insert(gap_i + 1, inferred_cx)
                                print(f"[pixel] missing bar inferred at position {gap_i+1} "
                                      f"cx={inferred_cx} "
                                      f"(gap {step:.0f} > 1.5× med {med_step:.0f})")
                                pixel_bar_count += 1
                                missing_inserted = True
                                break

                    if not missing_inserted:
                        print(f"[pixel] trimming month_labels from {num_bars} → {pixel_bar_count} (pixel count)")
                        month_labels = month_labels[-pixel_bar_count:]
                        num_bars     = pixel_bar_count

                    half_bar_w = max(3, cw // (num_bars * 3))

                # Check if one more bar is hiding at the right edge
                if len(bar_centers) >= 2:
                    avg_step = int(round(
                        sum(bar_centers[i+1] - bar_centers[i]
                            for i in range(len(bar_centers)-1))
                        / (len(bar_centers) - 1)
                    ))
                    extra_cx = bar_centers[-1] + avg_step
                    cs_e = max(0, extra_cx - half_bar_w)
                    ce_e = min(cw, extra_cx + half_bar_w)
                    if ce_e > cs_e:
                        col_e = is_bar[plot_top:plot_bot, cs_e:ce_e]
                        has_content = col_e.size > 0 and col_e.mean(axis=1).max() > COV_THRESH * 0.3
                        if has_content and extra_cx < cw:
                            try:
                                last_dt = datetime.strptime(month_labels[-1], "%b %y")
                                nm = last_dt.month % 12 + 1
                                ny = last_dt.year + (1 if last_dt.month == 12 else 0)
                                next_label = datetime(ny, nm, 1).strftime("%b %y")
                            except Exception:
                                next_label = "Next"
                            bar_centers.append(extra_cx)
                            month_labels.append(next_label)
                            num_bars += 1
                            print(f"[pixel] inferred right-edge bar cx={extra_cx} → {next_label}")

                print(f"[pixel] bar centers (pixel): {bar_centers}")
                bar_ranges = [
                    (max(0, cx - half_bar_w), min(cw, cx + half_bar_w))
                    for cx in bar_centers
                ]
            else:
                print(f"[pixel] pixel group count {pixel_bar_count} unreliable, using equal spacing")
                bar_span = bar_end - bar_start
                bar_step = bar_span / num_bars
                bar_ranges = [
                    (max(0, int(bar_start + (i + 0.5) * bar_step) - half_bar_w),
                     min(cw,  int(bar_start + (i + 0.5) * bar_step) + half_bar_w))
                    for i in range(num_bars)
                ]

        if not bar_ranges:
            bar_step   = cw / num_bars
            bar_ranges = [
                (max(0, int((i + 0.5) * bar_step) - half_bar_w),
                 min(cw,  int((i + 0.5) * bar_step) + half_bar_w))
                for i in range(num_bars)
            ]
            print(f"[pixel] bar region not found — equal spacing fallback (step={bar_step:.1f}px)")

        # --- 5. BAR-BOTTOM ABSOLUTE CALIBRATION ---
        # After we know bar x-positions, measure where bar-coloured pixels END
        # (bottom = x-axis = 0 kWh) in ABSOLUTE zoomed coords.  Because
        # abs_row = crop_z + abs_off is crop-shift-independent, the distance from
        # the top gridline to the x-axis is constant across LLM runs → stable slope.
        if not gridline_ok and gl_candidates and bar_ranges:
            bb_abs = []
            for cs_b, ce_b in bar_ranges:
                col_b = is_bar[plot_top:plot_bot, cs_b:ce_b]
                cov_b = col_b.mean(axis=1)
                bot_rows = np.where(cov_b > COV_THRESH * 0.5)[0]
                if len(bot_rows) >= 3:
                    bb_crop_z = float(bot_rows.max() + plot_top)
                    bb_abs.append(bb_crop_z + abs_off)

            if len(bb_abs) >= max(3, num_bars // 2):
                x_axis_abs  = float(np.median(bb_abs))
                top_gl_abs  = float((gl_candidates[0] + top) * SCALE)
                chart_range = x_axis_abs - top_gl_abs
                if chart_range > SCALE * 5:   # sanity: chart must be at least 5 unzoomed px
                    slope_bb     = -(y_max - y_min) / chart_range
                    intercept_bb = y_max - slope_bb * top_gl_abs
                    row_to_kwh   = (lambda row, s=slope_bb, i=intercept_bb, off=abs_off:
                                    float(s * (row + off) + i))
                    print(f"[pixel] bar-bottom calibration: x_axis_abs={x_axis_abs:.1f} "
                          f"top_gl_abs={top_gl_abs:.1f} range={chart_range:.1f}px "
                          f"slope={slope_bb:.4f}")

        # --- 6. EXTRACT BAR HEIGHTS ---
        # Uses center 60% of bar width (avoids anti-aliased edges),
        # gradient refinement of bar top, and sub-pixel interpolation.

        def get_bar_kwh(cs, ce, thresh=COV_THRESH):
            """Extract kWh for one bar column range at given coverage threshold."""
            col_slice = is_bar[plot_top:plot_bot, cs:ce]
            row_cov = col_slice.mean(axis=1)
            bar_rows = np.where(row_cov > thresh)[0]
            if len(bar_rows) == 0:
                return None, row_cov, None

            # Continuity check: find topmost sustained bar region
            min_run = max(3, (plot_bot - plot_top) // 40)
            bar_top_idx = None
            for row in bar_rows:
                end = min(len(row_cov), row + min_run)
                if np.sum(row_cov[row:end] > thresh) >= (end - row):
                    bar_top_idx = row
                    break
            if bar_top_idx is None:
                bar_top_idx = int(bar_rows.min())

            # Sub-pixel interpolation
            bar_top_float = float(bar_top_idx + plot_top)
            if bar_top_idx > 0:
                cov_above = float(row_cov[bar_top_idx - 1])
                cov_at    = float(row_cov[bar_top_idx])
                if cov_at > cov_above and (cov_at - cov_above) > 0:
                    t = (thresh - cov_above) / (cov_at - cov_above)
                    bar_top_float = float(plot_top + bar_top_idx - 1) + max(0.0, min(1.0, t))

            kwh = row_to_kwh(bar_top_float)
            kwh = round(max(y_min, min(y_max, kwh)))
            return int(kwh), row_cov, bar_top_float

        raw_kwh = []
        for i, label in enumerate(month_labels):
            cs, ce = bar_ranges[i]
            kwh, row_cov, bar_top_float = get_bar_kwh(cs, ce)
            if kwh is None:
                print(f"[pixel] {label}: no bar found")
            else:
                print(f"[pixel] {label}: bar_top={bar_top_float:.2f}px → {kwh} kWh")
            raw_kwh.append(kwh)

        # --- 6. OUTLIER RE-EXAMINATION ---
        # High outlier: bar >2x neighbors → retry with stricter threshold
        #   (avoids dark background pixels misread as bar content)
        # Low outlier: bar <20% of neighbor → retry with looser threshold
        #   (fixes inferred right-edge bar when fingerprint coverage is patchy)
        # Covers all bars including first and last (edge bars use single neighbor).
        for i in range(len(raw_kwh)):
            if raw_kwh[i] is None:
                continue
            neighbors = [raw_kwh[j] for j in (i - 1, i + 1)
                         if 0 <= j < len(raw_kwh) and raw_kwh[j] is not None]
            if not neighbors:
                continue
            neighbor_avg = sum(neighbors) / len(neighbors)
            if neighbor_avg <= 0:
                continue
            cs, ce = bar_ranges[i]

            if raw_kwh[i] > neighbor_avg * 2.0:
                print(f"[pixel] high outlier at {month_labels[i]}: {raw_kwh[i]} vs neighbor avg {neighbor_avg:.0f} — retrying")
                for retry_thresh in [COV_THRESH * 1.5, COV_THRESH * 2.0]:
                    kwh2, _, bar_top2 = get_bar_kwh(cs, ce, thresh=retry_thresh)
                    if kwh2 is not None and kwh2 <= neighbor_avg * 1.5:
                        print(f"[pixel] high outlier corrected: {raw_kwh[i]} → {kwh2} (thresh={retry_thresh:.2f})")
                        raw_kwh[i] = kwh2
                        break

            elif raw_kwh[i] < neighbor_avg * 0.20:
                print(f"[pixel] low outlier at {month_labels[i]}: {raw_kwh[i]} vs neighbor avg {neighbor_avg:.0f} — retrying")
                for retry_thresh in [COV_THRESH * 0.5, COV_THRESH * 0.3, COV_THRESH * 0.15]:
                    kwh2, _, bar_top2 = get_bar_kwh(cs, ce, thresh=retry_thresh)
                    if kwh2 is not None and kwh2 >= neighbor_avg * 0.5:
                        print(f"[pixel] low outlier corrected: {raw_kwh[i]} → {kwh2} (thresh={retry_thresh:.2f})")
                        raw_kwh[i] = kwh2
                        break

        # --- 7. TOTAL KWH CROSS-VALIDATION ---
        valid_vals = [v for v in raw_kwh if v is not None]
        if chart_total and chart_total > 0 and valid_vals:
            extracted_sum = sum(valid_vals)
            if extracted_sum > 0:
                scale_factor = chart_total / extracted_sum
                if 0.85 <= scale_factor <= 1.15:
                    print(f"[pixel] total validation: extracted={extracted_sum}, "
                          f"bill={chart_total}, scale={scale_factor:.4f}")
                    raw_kwh = [
                        round(v * scale_factor) if v is not None else None
                        for v in raw_kwh
                    ]
                else:
                    print(f"[pixel] total validation: scale={scale_factor:.3f} out of range, skipping")

        return [
            {"date": f"01 {label.upper()}", "kwh": int(raw_kwh[i]) if raw_kwh[i] is not None else None}
            for i, label in enumerate(month_labels)
        ]

    except Exception as e:
        print(f"[pixel_extract_bars] failed: {e}")
        import traceback; traceback.print_exc()
        return None


def extract_stacked_chart(pil_image, total_kwh, month_labels):
    """
    Reusable extractor for stacked / multi-colour MONTHLY bar charts with no
    printed numbers and faint gridlines (e.g. Tillsonburg's On/Mid/Off-peak,
    Milton's mixed Tiered + TOU bars).  Auto-detects the bar colours as the
    union of saturated NON-BLUE pixels (so a blue title bar / legend text is
    ignored), works for solid OR multi-colour-stacked bars, and:
      - finds the bar columns (drops the legend swatch on the right),
      - shared baseline (0 kWh) = median of column bottoms,
      - each bar's top = contiguous colour run up from the baseline,
      - calibrates scale from the known current-period total (newest bar = total).
    Returns [{"date","kwh"}] like pixel_extract_bars, or None.
    """
    try:
        import numpy as np
        a = np.array(pil_image.convert("RGB")).astype(int)
        H, W, _ = a.shape
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        maxd = np.maximum(np.maximum(abs(r - g), abs(g - b)), abs(r - b))
        bright = (r + g + b) / 3.0

        n = len(month_labels)
        if n == 0 or not total_kwh or total_kwh <= 0:
            print(f"[stacked] missing inputs: n_labels={n} total_kwh={total_kwh} "
                  f"— cannot calibrate (need current_period_kwh + month_labels)")
            return None

        # Bar pixels = coloured, but NOT blue (skips blue title bars / legend text).
        blueish = (b > r + 20) & (b > g + 20)
        bar = (maxd > 28) & (bright > 30) & (bright < 238) & ~blueish

        min_w = max(5, int(W * 0.006))
        gap_break = max(6, int(H * 0.025))

        col = bar.sum(0)
        on = col > max(8, col.max() * 0.10)
        groups = []
        s = None
        for x in range(W):
            if on[x] and s is None:
                s = x
            elif not on[x] and s is not None:
                groups.append((s, x - 1)); s = None
        if s is not None:
            groups.append((s, W - 1))

        groups = [gp for gp in groups if gp[1] - gp[0] >= min_w]
        if len(groups) < n:
            print(f"[stacked] only {len(groups)} bar columns, expected {n}")
            return None
        groups = sorted(groups)[:n]          # bars are leftmost; legend is right of them

        bottoms = []
        for (s, e) in groups:
            rows = np.where(bar[:, s:e+1].mean(1) > 0.3)[0]
            if len(rows):
                bottoms.append(int(rows.max()))
        if not bottoms:
            return None
        baseline = int(np.median(bottoms))

        def bar_top(s, e):
            cov = bar[:, s:e+1].mean(1) > 0.3
            top = baseline
            gap = 0
            for y in range(baseline, -1, -1):
                if cov[y]:
                    top = y; gap = 0
                else:
                    gap += 1
                    if gap >= gap_break:
                        break
            return top

        tops = [bar_top(s, e) for (s, e) in groups]
        newest_h = baseline - tops[-1]
        if newest_h <= 0:
            return None
        scale = total_kwh / newest_h
        print(f"[stacked] baseline={baseline} scale={scale:.2f} kWh/px "
              f"(newest bar = {total_kwh})")
        out = []
        for lab, t in zip(month_labels, tops):
            kwh = max(0, round((baseline - t) * scale))
            out.append({"date": f"01 {lab.upper()}", "kwh": kwh})
            print(f"[stacked] {lab}: top={t} → {kwh} kWh")
        return out
    except Exception as e:
        print(f"[stacked] failed: {e}")
        import traceback; traceback.print_exc()
        return None


_MONTH_DAYS = {"jan": 31, "feb": 28, "mar": 31, "apr": 30, "may": 31, "jun": 30,
               "jul": 31, "aug": 31, "sep": 30, "oct": 31, "nov": 30, "dec": 31}


def _conf_label(score):
    return "High" if score >= 90 else ("Medium" if score >= 75 else "Review")


def finalize_confidence(sig, monthly_history):
    """Turn the extraction signals + assembled 12-month history into an overall
    confidence score (+reasons) and per-month conf labels. See CONFIDENCE_SCORING.md.
    Confidence, NOT measured accuracy — see that doc for the caveats."""
    base = sig.get("base", 80)
    overall = base
    reasons = []
    method = sig.get("method")

    if method in ("printed", "printed_dailyavg"):
        reasons.append("✓ Read from numbers printed on the bill")
    elif method:
        reasons.append("⚠ History estimated by measuring chart bars (not independently verified)")

    if sig.get("agree") is False:
        overall -= 8
        reasons.append("⚠ The two extraction passes differed on a charge value")
    elif sig.get("verify_ran"):
        reasons.append("✓ Both extraction passes agreed on the charges")
    else:
        reasons.append("✓ Charges reconcile (components sum to the total)")

    if sig.get("anchor_known"):
        if sig.get("anchor_off"):
            overall -= 7
            reasons.append("⚠ Newest month doesn't match the bill's current-period total")
        else:
            reasons.append("✓ Newest month matches the bill's current-period total")

    if sig.get("tou_ok") is False:
        overall -= 5
        reasons.append("⚠ On/Mid/Off-peak don't sum to the total kWh")

    missing = sum(1 for e in (monthly_history or []) if e.get("kwh") is None)
    if missing:
        overall -= min(15, missing * 3)
        reasons.append(f"⚠ {missing} month(s) had no chart bar")

    overall = max(55, min(98, round(overall)))

    # Per-month confidence
    dated = [e for e in (monthly_history or []) if e.get("kwh") is not None]
    newest = (max(dated, key=lambda e: (e.get("year", 0), e.get("month_index", 0)))
              if dated else None)
    for e in (monthly_history or []):
        if e.get("kwh") is None:
            c = 40
        elif e is newest and sig.get("anchor_known"):
            c = 96
        elif e.get("days_fallback"):
            c = base - 12
        else:
            c = base
        e["conf"] = c
        e["conf_label"] = _conf_label(c)

    return {"overall": overall, "label": _conf_label(overall), "reasons": reasons[:4]}


def extract_dualbar_chart(pil_image, total_kwh, month_labels):
    """
    Reusable extractor for dual-bar "$ vs kWh" charts (e.g. Grand Bridge): each
    month has a DARK bar ($ amount, ignored) and a LIGHT-gray bar (kWh usage).
    Isolates the light usage bars, measures each as the longest SOLID colour run
    (so light-gray dashed gridlines / axis-label text don't count), and
    calibrates from the known current-period total (newest bar = total_kwh).
    Returns [{"date","kwh"}] or None.
    """
    try:
        import numpy as np
        a = np.array(pil_image.convert("RGB")).astype(int)
        H, W, _ = a.shape
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        bright = (r + g + b) / 3.0
        maxd = np.maximum(np.maximum(abs(r - g), abs(g - b)), abs(r - b))

        n = len(month_labels)
        if n == 0 or not total_kwh or total_kwh <= 0:
            print(f"[dualbar] missing inputs: n_labels={n} total_kwh={total_kwh}")
            return None

        # Two gray bar shades (dark $ ~55, light kWh ~190). Take the LIGHT one:
        # split gray pixels at the midpoint of their brightness range.
        gray = (maxd < 30) & (bright > 40) & (bright < 215)
        if gray.sum() < 200:
            return None
        gb = bright[gray]
        split = (float(np.percentile(gb, 20)) + float(np.percentile(gb, 80))) / 2
        light = (maxd < 30) & (bright > split) & (bright < 218)

        col = light.sum(0)
        on = col > max(6, col.max() * 0.20)
        groups = []
        s = None
        for x in range(W):
            if on[x] and s is None:
                s = x
            elif not on[x] and s is not None:
                groups.append((s, x - 1)); s = None
        if s is not None:
            groups.append((s, W - 1))
        groups = [gp for gp in groups if gp[1] - gp[0] >= max(5, int(W * 0.006))]
        if len(groups) < n:
            print(f"[dualbar] only {len(groups)} light bars, expected {n}")
            return None
        groups = sorted(groups)[:n]

        def longest_run(s, e):
            cov = light[:, s:e + 1].mean(1) > 0.5
            best = (0, 0, 0); cur = None
            for y in range(H):
                if cov[y]:
                    if cur is None:
                        cur = y
                elif cur is not None:
                    if y - cur > best[2]:
                        best = (cur, y - 1, y - cur)
                    cur = None
            if cur is not None and H - cur > best[2]:
                best = (cur, H - 1, H - cur)
            return best[0], best[1]

        runs = [longest_run(s, e) for (s, e) in groups]
        baseline = int(np.median([bot for _, bot in runs]))
        tops = [t for t, _ in runs]
        newest_h = baseline - tops[-1]
        if newest_h <= 0:
            return None
        scale = total_kwh / newest_h
        print(f"[dualbar] baseline={baseline} split={split:.0f} scale={scale:.3f} "
              f"(newest bar = {total_kwh})")
        out = []
        for lab, t in zip(month_labels, tops):
            kwh = max(0, round((baseline - t) * scale))
            out.append({"date": f"01 {lab.upper()}", "kwh": kwh})
            print(f"[dualbar] {lab}: top={t} → {kwh} kWh")
        return out
    except Exception as e:
        print(f"[dualbar] failed: {e}")
        import traceback; traceback.print_exc()
        return None


def extract_dailyavg_chart(pil_image, month_labels, gridline_values):
    """
    Reusable extractor for single-colour bar charts whose y-axis is a DAILY
    AVERAGE ("kWh per day", e.g. Guelph / Alectra).  Auto-detects the bar colour,
    calibrates from the chart's own (crisp) gridlines, reads each bar's daily
    average, then multiplies by the number of days in that month to return
    monthly kWh.  gridline_values = labelled y-axis values (only the interval is
    used).  Returns [{"date","kwh"}] or None.
    """
    try:
        import numpy as np
        a = np.array(pil_image.convert("RGB")).astype(int)
        H, W, _ = a.shape
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        bright = (r + g + b) / 3.0

        n = len(month_labels)
        vals = sorted(set(gridline_values or []), reverse=True)   # [150,120,..,0]
        k = len(vals)
        if n == 0 or k < 2:
            print(f"[dailyavg] missing inputs: n_labels={n} gridlines={k}")
            return None

        # Detect dark horizontal gridline rows.
        dark = bright < 100
        cov = dark[:, int(W * 0.15):int(W * 0.95)].mean(1)
        gl_rows = [y for y in range(H) if cov[y] > 0.35]
        gg = []
        for y in gl_rows:
            if not gg or y - gg[-1][-1] > 4:
                gg.append([y])
            else:
                gg[-1].append(y)
        grid = sorted(int(np.mean(c)) for c in gg)
        if len(grid) < k:
            print(f"[dailyavg] found {len(grid)} gridlines, need {k}")
            return None

        # Pick the consecutive run of k gridlines with the most even spacing
        # (drops a title line above or an axis border below).
        best = None
        for i in range(len(grid) - k + 1):
            win = grid[i:i + k]
            d = np.diff(win)
            score = float(np.std(d) / (np.mean(d) + 1e-6))
            if best is None or score < best[0]:
                best = (score, win)
        win = best[1]
        top_row, zero_row = win[0], win[-1]
        coeffs = np.polyfit(win, vals, 1)          # row → y-axis value (daily avg)

        # Bar fill colour = dominant non-white/non-black pixel in the plot band.
        band = a[top_row:zero_row + 1, :, :].reshape(-1, 3)
        bbr = band.mean(1)
        fill = band[(bbr > 55) & (bbr < 232)]
        if len(fill) < 200:
            return None
        bc = [int(np.median(fill[:, 0])), int(np.median(fill[:, 1])), int(np.median(fill[:, 2]))]
        bar = np.sqrt((r - bc[0])**2 + (g - bc[1])**2 + (b - bc[2])**2) < 70
        # Gridlines can be the same gray as the bars — remove them so they don't
        # read as bar tops spanning every column.
        for gy in grid:
            bar[max(0, gy - 1):gy + 2, :] = False
        barband = np.zeros_like(bar)
        barband[top_row:zero_row + 1, :] = bar[top_row:zero_row + 1, :]

        # Bar columns within the plot band.
        col = barband.sum(0)
        on = col > max(4, col.max() * 0.06)
        groups = []
        s = None
        for x in range(W):
            if on[x] and s is None:
                s = x
            elif not on[x] and s is not None:
                groups.append((s, x - 1)); s = None
        if s is not None:
            groups.append((s, W - 1))
        groups = [gp for gp in groups if gp[1] - gp[0] >= max(5, int(W * 0.008))]

        def toprow(s, e):
            ys = np.where(barband[:, s:e + 1].mean(1) > 0.10)[0]
            return int(ys.min()) if len(ys) else None
        scored = [(s, e, toprow(s, e)) for (s, e) in groups]
        scored = [t for t in scored if t[2] is not None]
        if len(scored) < n:
            print(f"[dailyavg] only {len(scored)} bar columns, expected {n}")
            return None
        if len(scored) > n:                 # drop shortest extras (axis-label sliver)
            scored.sort(key=lambda t: zero_row - t[2], reverse=True)
            scored = scored[:n]
        scored.sort(key=lambda t: t[0])

        print(f"[dailyavg] color={bc} gridlines→{win} (top={vals[0]} zero_row={zero_row})")
        out = []
        for lab, (s, e, top) in zip(month_labels, scored):
            daily = float(np.polyval(coeffs, top))
            days = _MONTH_DAYS.get(lab.strip()[:3].lower(), 30)
            kwh = max(0, round(daily * days))
            out.append({"date": f"01 {lab.upper()}", "kwh": kwh})
            print(f"[dailyavg] {lab}: {daily:.1f}/day × {days}d → {kwh} kWh")
        return out
    except Exception as e:
        print(f"[dailyavg] failed: {e}")
        import traceback; traceback.print_exc()
        return None


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
    """Majority-vote the PRINTED usage-history kWh values across independent reads.

    Printed numbers are ground truth, but a single transcription occasionally
    digit-slips (e.g. reads 1903 as 1685). Slips are uncorrelated between reads,
    so we re-transcribe the history a second time and, only if it disagrees with
    the first pass, a third time, then take the per-row majority. Agreement after
    two reads early-exits (one extra API call in the common case).

    base_history: [{date, kwh}] from the main extraction pass (this is vote 1).
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

    # The base read comes from the whole page and tends to ESTIMATE these values
    # from bar length, so it's the least reliable source for printed digits. The
    # focused re-reads look at the zoomed number column, where there are no bars to
    # estimate from. So we take THREE focused re-reads and majority-vote among them;
    # the base read is only a fallback for rows the re-reads don't return.
    rr_votes = {}
    for temp in (0.4, 0.7, 1.0):
        _tally(_reread(temp), rr_votes)
    base_votes = {}
    _tally(base_history, base_votes)

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
    Extract bill data from images using Claude.
    If pil_images provided and chart has no printed numbers,
    uses pixel analysis for monthly_usage_history instead of Claude estimation.
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

    # --- Chart meta pass: determine if pixel analysis is needed ---
    pixel_history = None
    chart_meta = {}           # chart geometry (crop box, page) from the meta pass
    chart_via_total = False   # True when a calibrate-by-total extractor was used
    _cur_period = None        # confidence: bill's current-period total (anchor check)
    _ext_type = None          # confidence: which dedicated extractor ran
    _has_printed = False      # chart prints its kWh values as text (consensus path)
    if pil_images:
        try:
            meta_response = client.messages.create(
                model=META_MODEL,
                max_tokens=600,
                messages=[{
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": CHART_META_PROMPT}]
                }]
            )
            chart_meta = clean_json(meta_response.content[0].text)
            chart_meta = apply_chart_profile(chart_meta)
            _cur_period = chart_meta.get("current_period_kwh")
            _ext_type = chart_meta.get("extractor")
            _has_printed = bool(chart_meta.get("has_printed_numbers"))
            print(f"[chart meta] has_bar_chart={chart_meta.get('has_bar_chart')} "
                  f"has_printed_numbers={chart_meta.get('has_printed_numbers')} "
                  f"page_index={chart_meta.get('page_index')} "
                  f"issuer={chart_meta.get('issuer')!r} "
                  f"current_period_kwh={chart_meta.get('current_period_kwh')} "
                  f"n_labels={len(chart_meta.get('month_labels') or [])} "
                  f"extractor={chart_meta.get('extractor')}")

            if chart_meta.get("has_bar_chart") and not chart_meta.get("has_printed_numbers", True):
                page_idx = min(chart_meta.get("page_index", 0), len(pil_images) - 1)
                ext = chart_meta.get("extractor")
                if ext:
                    # Dedicated extractor. Claude's page_index is unreliable when
                    # the chart is a separate file, so try every page and take
                    # whichever actually yields the full chart.
                    labels = chart_meta.get("month_labels") or []
                    for pi in range(len(pil_images)):
                        if ext == "stacked":
                            ph = extract_stacked_chart(
                                pil_images[pi],
                                chart_meta.get("current_period_kwh"), labels)
                        elif ext == "dailyavg":
                            ph = extract_dailyavg_chart(
                                pil_images[pi], labels,
                                chart_meta.get("y_axis_gridlines"))
                        elif ext == "dualbar":
                            ph = extract_dualbar_chart(
                                pil_images[pi],
                                chart_meta.get("current_period_kwh"), labels)
                        else:
                            ph = None
                        if ph and len(ph) == len(labels) and len(labels) > 0:
                            pixel_history = ph
                            chart_via_total = True   # self-calibrated; skip offset-anchor
                            print(f"[route] {ext} chart found on page {pi}")
                            break
                else:
                    pixel_history = pixel_extract_bars(pil_images[page_idx], chart_meta)
                if pixel_history:
                    print(f"[pixel analysis] extracted {len(pixel_history)} bars")

        except Exception as e:
            print(f"[chart meta pass] failed: {e}")

    # --- Pass 1: main extraction ---
    response1 = client.messages.create(
        model=EXTRACT_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": image_content + [{"type": "text", "text": EXTRACT_PROMPT}]
        }]
    )
    data = clean_json(response1.content[0].text)

    # Printed-number history: the kWh values are printed as text (ground truth), but
    # a single read occasionally digit-slips. Re-transcribe and majority-vote the
    # history so those slips can't reach the table. Pixel/estimated charts skip this
    # (they have their own dedicated extractors and aren't transcriptions).
    if _has_printed and not pixel_history and (data.get("monthly_usage_history") or []):
        focus_b64 = None
        if pil_images:
            pidx = min(chart_meta.get("page_index", 0) or 0, len(pil_images) - 1)
            box = locate_history_table(client, image_content)
            if not box:
                # Locate failed/rejected — fall back to a known per-issuer box.
                issuer = (chart_meta.get("issuer") or "").lower()
                for name, b in HISTORY_TABLE_BOXES.items():
                    if name in issuer:
                        box = b
                        print(f"[history-consensus] using profile box for '{name}'")
                        break
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

    # Daily-average charts (e.g. Alectra): Claude reads the raw daily averages and
    # each bar's "# of days"; we convert to monthly totals here deterministically
    # (the model won't reliably do the multiplication itself).
    if data.get("history_is_daily_average") and not pixel_history:
        hist = data.get("monthly_usage_history") or []
        cal = list(_MONTH_DAYS.values())   # calendar days indexed by month 0-11
        # Sanity guard: real daily averages are small (~5-200/day). If the values are
        # already in the hundreds/thousands, they're MONTHLY totals despite a "kWh/day"
        # axis label (e.g. Toronto prints monthly kWh next to a daily-average chart) —
        # multiplying by days would ~30x them. Treat as already-monthly and skip.
        raws = sorted(e["kwh"] for e in hist if isinstance(e.get("kwh"), (int, float)))
        median_raw = raws[len(raws) // 2] if raws else 0
        if median_raw > 200:
            print(f"[daily-avg] values already monthly-scale (median {median_raw:.0f}); "
                  f"NOT multiplying by days")
            data["history_is_daily_average"] = False
            hist = []   # skip the conversion loop below
        conv = 0
        for e in hist:
            if e.get("kwh") is None:
                continue
            days = e.get("days")
            # Days guardrail: billing periods are ~28-34 days. An out-of-range read
            # (e.g. 23) is almost certainly an OCR slip → fall back to calendar days.
            if not (isinstance(days, (int, float)) and 25 <= days <= 35):
                _, mi = parse_history_date(e.get("date", ""))
                days = cal[mi] if mi is not None else 30
                e["days_fallback"] = True   # confidence: OCR day-count was implausible
            e["kwh"] = round(e["kwh"] * days)
            conv += 1
        # Anchor the newest bar to the bill's exact current-period total (the most
        # recent bar IS the current billing period, whose total the bill prints).
        total = data.get("total_kwh")
        dated = [(parse_history_date(e.get("date", "")), e)
                 for e in hist if e.get("kwh") is not None]
        dated = [(d, e) for d, e in dated if d[0] is not None]
        if total and dated:
            newest = max(dated, key=lambda x: x[0])[1]
            newest["kwh"] = round(total)
        if conv:
            print(f"[daily-avg] converted {conv} bars (days guardrail; "
                  f"newest anchored to total_kwh={total})")

    # Override monthly_history with pixel results if available
    if pixel_history:
        data["monthly_usage_history"] = pixel_history

    # --- Pass 2: verification (conditional) ---
    # A second model re-read mainly catches digit slips in the charges. When pass 1
    # is already internally consistent — the total was read AND the TOU/tier
    # components sum to it — that's a strong signal it's right, so skip the extra
    # call to save time. Re-verify only when the total is missing or the parts
    # don't reconcile.
    _total = data.get("total_kwh")
    _bt = (data.get("bill_type") or "").upper()
    if "TIER" in _bt:
        _parts = [data.get(k) for k in ("tier1_kwh", "tier2_kwh")]
    else:
        _parts = [data.get(k) for k in
                  ("on_peak_kwh", "mid_peak_kwh", "off_peak_kwh", "overnight_kwh")]
    _parts = [p for p in _parts if isinstance(p, (int, float))]
    if not _total or not _parts:
        needs_verify = True
    else:
        needs_verify = abs(sum(_parts) - _total) > 0.02 * max(_total, 1)

    verify_ran = False
    if needs_verify:
        verify_ran = True
        if pixel_history:
            verify_note = (
                "\n\nIMPORTANT: monthly_usage_history was measured by pixel analysis and is accurate. "
                "Do NOT change it. Only verify the other numeric fields."
            )
        elif data.get("history_is_daily_average"):
            verify_note = (
                "\n\nIMPORTANT: monthly_usage_history has already been converted from daily "
                "averages to MONTHLY totals (daily average × number of days) and is correct. "
                "Do NOT change the monthly_usage_history values. Only verify the other fields."
            )
        else:
            verify_note = ""

        verify_prompt = VERIFY_PROMPT.format(data=json.dumps(data, indent=2)) + verify_note
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

    # Always re-inject pixel history in case Claude changed it despite the note
    if pixel_history:
        verified["monthly_usage_history"] = pixel_history
    elif data.get("history_is_daily_average"):
        # Use the code-converted (daily × days) history; ignore any change verify made.
        verified["monthly_usage_history"] = data.get("monthly_usage_history")

    # --- ANCHOR CHART TO KNOWN CURRENT-PERIOD TOTAL ---
    # The newest chart bar is the current billing period, whose kWh we read
    # directly from the bill (total_kwh).  Evenly-spaced gridlines can make the
    # pixel calibration lock onto the wrong line as zero, shifting EVERY bar by
    # a constant (~one grid interval — e.g. Burlington read ~97 kWh low across
    # the board).  If the newest bar is off from the known total by a plausible
    # *systematic* amount, correct all bars by that constant offset.
    if pixel_history and verified.get("total_kwh") and not chart_via_total:
        hist = verified["monthly_usage_history"]
        anchor = verified["total_kwh"]
        last = hist[-1]["kwh"] if hist and hist[-1].get("kwh") is not None else None
        if last and anchor and 0.03 * anchor < abs(anchor - last) <= 0.25 * anchor:
            offset = anchor - last
            for e in hist:
                if e.get("kwh") is not None:
                    e["kwh"] = max(0, round(e["kwh"] + offset))
            print(f"[anchor] shifted chart by {offset:+.0f} kWh "
                  f"(newest bar {last}→{anchor} to match bill total)")

    # --- CONFIDENCE SIGNALS (see CONFIDENCE_SCORING.md) ---
    def _close(a, b, tol=0.02):
        if a is None or b is None:
            return True
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return True
        if max(abs(a), abs(b)) == 0:
            return True
        return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)

    hist = verified.get("monthly_usage_history") or []
    if not hist:
        method, base = None, 87                         # charges only, no history chart
    elif pixel_history:
        if _ext_type in ("stacked", "dualbar"):
            method, base = "pixel_anchor", 85
        else:
            method, base = "pixel_gridline", 80
    elif data.get("history_is_daily_average"):
        method, base = "printed_dailyavg", 90
    else:
        method, base = "printed", 93

    agree = all(_close(data.get(k), verified.get(k)) for k in
                ("total_kwh", "on_peak_kwh", "mid_peak_kwh", "off_peak_kwh", "overnight_kwh"))

    cur = _cur_period or verified.get("total_kwh")
    dd = [(parse_history_date(e.get("date", "")), e.get("kwh"))
          for e in hist if e.get("kwh") is not None]
    dd = [(d, v) for d, v in dd if d[0] is not None]
    newest_val = max(dd, key=lambda x: x[0])[1] if dd else None
    # Anchor check only validates PIXEL extractions; printed numbers are read
    # exactly, so a chart-vs-bill period mismatch shouldn't penalize them.
    anchor_known = bool(cur and newest_val and method in ("pixel_anchor", "pixel_gridline"))
    anchor_off = bool(anchor_known and not _close(newest_val, cur, tol=0.03))

    parts = [verified.get(k) for k in
             ("on_peak_kwh", "mid_peak_kwh", "off_peak_kwh", "overnight_kwh")]
    parts = [p for p in parts if p is not None]
    tou_ok = True
    if parts and verified.get("total_kwh"):
        tou_ok = _close(sum(parts), verified.get("total_kwh"), tol=0.03)

    verified["_conf_signals"] = {
        "method": method, "base": base, "agree": agree, "verify_ran": verify_ran,
        "anchor_known": anchor_known, "anchor_off": anchor_off, "tou_ok": tou_ok,
    }

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

        # Confidence score (also tags each month in monthly_history with conf/conf_label).
        confidence = finalize_confidence(data.get("_conf_signals") or {}, monthly_history)

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
            "confidence": confidence,
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

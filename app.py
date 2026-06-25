import os
import io
import re
import json
import base64
import uuid
from itertools import combinations
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
- monthly_usage_history = ALL bars from the usage history chart (typically 13-15 entries).
  * Format every date as "DD MMM YY" (e.g. "26 FEB 26"). If dates are shown as "26-Feb-26", convert to "26 Feb 26".
  * Use the billing period END date (read date) as each entry's date.
  * If the Y-axis label says "KWH per day" or "Daily Average": look for a "# of Days" row below the chart. For each bar, compute monthly_kwh = round(daily_avg × days). Return the computed total kWh — NOT the raw daily average.
  * If the Y-axis shows monthly kWh totals: return the bar values directly.
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
  "chart_total_kwh": number or null
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


def file_to_images(file_path):
    """
    Convert a PDF or PNG to (pil_images, b64_strings).
    pil_images = original PIL images (for pixel analysis).
    b64_strings = enhanced base64 strings (for Claude OCR).
    """
    pil_images = []
    b64_strings = []

    if file_path.lower().endswith(".png"):
        original = Image.open(file_path).convert("RGB")
        enhanced = enhance_image(original.copy())
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        pil_images.append(original)
        b64_strings.append(base64.standard_b64encode(buf.getvalue()).decode("utf-8"))
    else:
        pages = convert_from_path(file_path, dpi=300)
        for page in pages:
            original = page.convert("RGB")
            enhanced = enhance_image(original.copy())
            buf = io.BytesIO()
            enhanced.save(buf, format="PNG")
            pil_images.append(original)
            b64_strings.append(base64.standard_b64encode(buf.getvalue()).decode("utf-8"))

    return pil_images, b64_strings


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

            for gl_thresh in [0.50, 0.40, 0.30, 0.20, 0.12]:
                cands = cluster_rows(np.where(gray_cov_o > gl_thresh)[0], gap=3, weights=gray_cov_o)
                print(f"[pixel] gridline thresh={gl_thresh}: {len(cands)} candidates "
                      f"(need {len(gl_values)}): rows={[round(c,1) for c in cands]}")
                if not all_gray_cands and len(cands) >= 2:
                    all_gray_cands = cands
                if len(cands) == len(gl_values):
                    gl_candidates = cands
                    gl_px   = np.array([row * SCALE for row in gl_candidates], dtype=float)
                    gl_vals = np.array(gl_values, dtype=float)
                    coeffs  = np.polyfit(gl_px, gl_vals, 1)
                    row_to_kwh = lambda row, c=coeffs: float(np.polyval(c, row))
                    print(f"[pixel] gridline calibration OK slope={coeffs[0]:.4f} intercept={coeffs[1]:.1f}")
                    break

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
                        gl_px   = np.array([row * SCALE for row in gl_candidates], dtype=float)
                        gl_vals = np.array(gl_values, dtype=float)
                        coeffs  = np.polyfit(gl_px, gl_vals, 1)
                        row_to_kwh = lambda row, c=coeffs: float(np.polyval(c, row))
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
                            gl_px  = np.array([r * SCALE for r in new_cands], dtype=float)
                            gl_v   = np.array(trimmed, dtype=float)
                            c      = np.polyfit(gl_px, gl_v, 1)
                            res    = np.abs(np.polyval(c, gl_px) - gl_v).max()
                            if res < 15:
                                gl_candidates = list(new_cands)
                                row_to_kwh = lambda row, cf=c: float(np.polyval(cf, row))
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
                        px_sub = np.array([row * SCALE for row in cand_sub], dtype=float)
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
                    row_to_kwh = lambda row, c=best_coeffs: float(np.polyval(c, row))
                    print(f"[pixel] partial gridline fit: {len(best_cands)} rows, "
                          f"slope={best_coeffs[0]:.4f}, max_residual={best_res:.1f}")
                else:
                    print(f"[pixel] gridline calibration failed (best residual={best_res:.1f}) "
                          f"— falling back to floor detection")
            elif row_to_kwh is None:
                print("[pixel] gridline calibration failed — falling back to floor detection")

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

        # --- 4. BAR COLOR SELECTION ---
        # Priority: Claude fingerprint > auto-detected > generic
        # Prefer whichever is most selective while still covering enough pixels.
        gen_count = int(is_bar_generic[plot_top:plot_bot, :].sum())

        if bar_color and len(bar_color) == 3:
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

        # --- 5. EXTRACT BAR HEIGHTS ---
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
    if pil_images:
        try:
            meta_response = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=600,
                messages=[{
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": CHART_META_PROMPT}]
                }]
            )
            chart_meta = clean_json(meta_response.content[0].text)

            if chart_meta.get("has_bar_chart") and not chart_meta.get("has_printed_numbers", True):
                page_idx = min(chart_meta.get("page_index", 0), len(pil_images) - 1)
                pixel_history = pixel_extract_bars(pil_images[page_idx], chart_meta)
                if pixel_history:
                    print(f"[pixel analysis] extracted {len(pixel_history)} bars")

        except Exception as e:
            print(f"[chart meta pass] failed: {e}")

    # --- Pass 1: main extraction ---
    response1 = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": image_content + [{"type": "text", "text": EXTRACT_PROMPT}]
        }]
    )
    data = clean_json(response1.content[0].text)

    # Override monthly_history with pixel results if available
    if pixel_history:
        data["monthly_usage_history"] = pixel_history

    # --- Pass 2: verification ---
    if pixel_history:
        verify_note = (
            "\n\nIMPORTANT: monthly_usage_history was measured by pixel analysis and is accurate. "
            "Do NOT change it. Only verify the other numeric fields."
        )
    else:
        verify_note = ""

    verify_prompt = VERIFY_PROMPT.format(data=json.dumps(data, indent=2)) + verify_note
    response2 = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": image_content + [{"type": "text", "text": verify_prompt}]
        }]
    )
    raw2 = response2.content[0].text
    print(f"[verify] response ({len(raw2)} chars): {raw2[:200]}")
    verified = clean_json(raw2)

    # Always re-inject pixel history in case Claude changed it despite the note
    if pixel_history:
        verified["monthly_usage_history"] = pixel_history

    return verified


def parse_history_date(date_str):
    """Parse '20 MAR 26', '20 MAR 2026', or '26-Feb-26' → (year, month_index 0-11)."""
    for fmt in ("%d %b %y", "%d %b %Y", "%d-%b-%y", "%d-%b-%Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.year, dt.month - 1
        except ValueError:
            continue
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
        year, month_idx = parse_history_date(entry.get("date", ""))
        if year is not None:
            parsed.append({
                "year": year,
                "month_index": month_idx,
                "month_abbr": MONTH_ABBR[month_idx],
                "kwh": entry.get("kwh")
            })

    if not parsed:
        return []

    parsed.sort(key=lambda x: (x["year"], x["month_index"]))

    # Determine window endpoint: prefer billing_period_end over last data point.
    anchor_y, anchor_m = None, None
    if billing_period_end:
        try:
            dt = datetime.strptime(billing_period_end.strip(), "%m/%d/%Y")
            anchor_y, anchor_m = dt.year, dt.month - 1
        except Exception:
            pass

    if anchor_y is not None:
        end_y, end_m = anchor_y, anchor_m
        # Keep only entries within the 12-month window ending at billing period end.
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

    # Group files into bills:
    #   - Each PDF is its own bill (already multi-page internally).
    #   - PNGs sharing the same filename prefix (digits stripped from the end)
    #     are treated as pages of one bill, e.g. "hydro1.png" + "hydro2.png"
    #     → one bill. "jan.png" + "feb.png" → two bills (different prefixes).
    groups = {}  # key → list of FileStorage objects, preserving upload order
    group_order = []

    for f in files:
        ext = f.filename.lower().rsplit(".", 1)[-1]
        if ext not in ("pdf", "png"):
            continue
        if ext == "pdf":
            key = f.filename          # PDFs never share a group
        else:
            base = f.filename.rsplit(".", 1)[0]
            key = re.sub(r"\d+$", "", base) or base   # strip trailing digits
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(f)

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

        month, year = month_from_date(data.get("billing_period_end"))
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

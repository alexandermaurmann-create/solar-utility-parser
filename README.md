# Solar Utility Bill Parser

Web app that extracts TOU (time-of-use) data from utility bill PDFs and outputs a clean table ready to paste into Excel.

## What it extracts

- On Peak / Mid Peak / Off Peak kWh
- Total monthly kWh
- Delivery Charge ($)
- Regulatory Charge ($)
- Billing period start & end dates
- Organized January → December (most recent 12 months)

---

## Deploy to Render via GitHub

### 1. Push to GitHub
Create a new GitHub repo and push this folder:
```bash
cd solar-utility-parser
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/solar-utility-parser.git
git push -u origin main
```

### 2. Create Render Web Service
1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Render will auto-detect `render.yaml` and configure everything
4. Click **Deploy** — done!

Your app will be live at `https://solar-utility-parser.onrender.com` (or similar).

---

## Usage

1. Open the app URL in any browser
2. Drag & drop PDF utility bills (one per billing month, up to 12)
3. Click **Parse Bills**
4. Click **Copy for Excel** → paste directly into a spreadsheet
   — or click **Download CSV** to save the file

---

## Notes

- Works best with text-based PDFs (not scanned images)
- Upload all 12 months at once for the full year view
- Fields that can't be found in a PDF show as "—"

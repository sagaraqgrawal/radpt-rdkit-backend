# RADPT RDKit Backend — Deployment Guide

This is a small Python API that computes **real** molecular descriptors with RDKit
(molecular weight, LogP, TPSA, QED, Lipinski/Veber/Egan/Ghose/Muegge filters,
PAINS/Brenk structural alerts, and more). RADPT calls it so those numbers are
genuinely *calculated from structure*, not AI-estimated.

It runs separately from your Cloudflare AI Worker. The AI Worker still handles the
ADMET/tox predictions RDKit cannot compute (hERG, DILI, CYP, clearance, etc.).

--------------------------------------------------------------------------------
## Files in this folder
- `app.py`            — the Flask + RDKit API
- `requirements.txt`  — Python dependencies
- `render.yaml`       — Render blueprint (optional one-click config)
- `README.md`         — this guide

--------------------------------------------------------------------------------
## Deploy on Render.com (free, ~5 minutes)

### Option A — via GitHub (recommended)
1. Create a free GitHub account if you don't have one.
2. Make a new repository, e.g. `radpt-rdkit-backend`.
3. Upload these three files to it: `app.py`, `requirements.txt`, `render.yaml`.
4. Go to https://render.com → sign up (free) → **New +** → **Web Service**.
5. Connect your GitHub and pick the `radpt-rdkit-backend` repo.
6. Render auto-detects Python. Confirm these settings (render.yaml sets them too):
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
   - **Instance type:** Free
7. Click **Create Web Service**. First build takes ~3-5 min (RDKit is a big wheel).
8. When it's live you'll get a URL like:
   `https://radpt-rdkit.onrender.com`
9. Test it in your browser — visiting that URL should show:
   `{"status":"ok","service":"RADPT RDKit backend","rdkit":"2024.03.5"}`

### Option B — without GitHub
Render also supports deploying from a public Git URL or via their CLI; GitHub is
simplest. If you prefer, zip these files and use Render's "Deploy from Git" flow.

--------------------------------------------------------------------------------
## Connect it to RADPT
1. Copy your live Render URL (e.g. `https://radpt-rdkit.onrender.com`).
2. Open `radpt.html`, find the line near the top of the main script:
       const RDKIT_URL = "https://YOUR-RDKIT-SERVICE.onrender.com";
   Replace it with your real URL. **No trailing slash.**
3. Re-upload `radpt.html` to radgent.com. Done.

--------------------------------------------------------------------------------
## Important note about Render's free tier
Free services **sleep after ~15 minutes of inactivity** and take ~30-50 seconds to
wake on the next request. So the first RDKit call after idle time will be slow; RADPT
handles this gracefully (it shows a "waking calculation engine…" state and falls back
to AI values if the backend is still asleep). If you want it always-on, Render's
paid tier removes the sleep, or you can ping the health URL every few minutes with a
free uptime service like UptimeRobot.

--------------------------------------------------------------------------------
## Security (optional but recommended)
In `app.py`, the CORS line currently allows any origin (`"*"`). Once it works, lock it
to your site:
    CORS(app, resources={r"/*": {"origins": "https://radgent.com"}})
Then redeploy.

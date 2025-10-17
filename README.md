# **ZenTej eKYC: Deepfake-Proof Identity Verification**
### Real-time eKYC with liveness & deepfake defense

**Real-time eKYC web application** with **deepfake/anti-spoofing** and **face matching**.  
Built with **Next.js + Tailwind + shadcn/ui + React Three Fiber (three.js)**, **Supabase** (Auth/DB/RLS), and a lightweight **FastAPI** inference service that runs two PyTorch models in sequence: **Forgery** (liveness/anti-spoof) → **Identification** (embeddings).



<img width="1416" height="770" alt="Screenshot 2025-10-18 at 12 43 56 AM" src="https://github.com/user-attachments/assets/d046bfb2-f9f2-4c99-b45d-751061bfb570" />



## **Highlights**
- **Two-model pipeline:** Forgery check runs **first**; Identification runs **only if real**.  
- **Three deterministic login outcomes (exact text):**  
  1. **Fake image detected.**  
  2. **No match found for this account.** *(1:1)* / **No match found in the database.** *(1:N)*  
  3. **Login successful.**  
- **Hackathon I/O contract:**  
  **Input:** two facial images **or** a short selfie video (8–24 frames).  
  **Output:** **Match Score**, **Liveness Score**, **Authenticity Label** (`authentic`/`fake`).  
- **Privacy-first:** AES-GCM encrypted embeddings; **no raw images stored**.  
- **Strict RLS** on Supabase; clean, accessible **three.js** hero with reduced-motion fallback.


## **Architecture**
apps/
  web/          # Next.js app (UI + thin API routes calling Supabase & FastAPI)
  inference/    # FastAPI service (preprocess → forgery → identification)
supabase/
  migrations/   # SQL schema + RLS (+ optional pgvector)

**Supabase** → single source of truth (users, encrypted templates, audit log).
**FastAPI**  → face detection/alignment + two-model inference + final decision.
**Next.js**  → capture UI, calls FastAPI via server routes, renders scores/outcomes.

---

## **Deployment Guide — ZenTej eKYC (Local + Cloud + Quantization + Railway)**

### 1) **Prerequisites**
- Node.js ≥ 18 and **pnpm** ≥ 8  
- Python ≥ 3.10 (virtualenv recommended)  
- A **Supabase** project (Auth + Postgres enabled)  
- Two trained model weight files available (do **not** commit large binaries)  

---

### 2) **Supabase Setup (once)**
- Create a new Supabase project.  
- Run schema & RLS migrations (tables: `profiles`, `face_templates`, `verification_events`, `threshold_config`; optional `pgvector`).  
- Copy credentials for the web env:
  - `NEXT_PUBLIC_SUPABASE_URL`
  - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
  - `SUPABASE_SERVICE_ROLE_KEY`

---

### 3) **Environment Variables**
**Web app (Next.js)**
- `NEXT_PUBLIC_SUPABASE_URL`  
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`  
- `SUPABASE_SERVICE_ROLE_KEY`  *(server only)*  
- `INFERENCE_API_URL`  *(public URL of the inference service, e.g., https://zen-tej-api.up.railway.app)*  
- `ENCRYPTION_KEY_HEX`  *(32-byte hex for AES-GCM at rest)*  

**Inference service (FastAPI)**
- `MODEL_ID_PATH`  *(identification weights path)*  
- `MODEL_F_PATH`   *(forgery/liveness weights path)*  
- `MODEL_ID_FORMAT` (`torch`|`onnx`)  
- `MODEL_F_FORMAT`  (`torch`|`onnx`)  
- `INPUT_SIZE` (`224`)  
- `NORM_SCHEME` (`minus_one_to_one`)  
- `DEVICE` (`auto` / `cpu` / `cuda:0`)  
- `THRESH_MATCH`, `THRESH_FAKE`, `THRESH_LIVE`  
- `LOGIN_SEARCH_MODE` (`1to1`|`1toN`)  
- `DUMMY_MODE` (`true`|`false`)  
- `FASTAPI_PORT` (use `$PORT` on Railway)  

---

### 4) **Install Dependencies**
~~~bash
# Node workspace
pnpm i

# Python service (inside your virtualenv)
pip install -r requirements.txt
~~~

---

### 5) **Quantization (reduce latency & size)**
**Goal:** run lighter, faster inference while keeping accuracy acceptable for demo. Choose one path per model.

A) **PyTorch → Dynamic INT8 (quick, CPU-friendly)**  
- Works well for MLP/Linear layers; no calibration data needed.
~~~python
# scripts/quantize_dynamic_torch.py
import torch
from torch.ao.quantization import quantize_dynamic

model = torch.load("apps/inference/models/id_model.pth", map_location="cpu")
qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(qmodel, "apps/inference/models/id_model_q.pth")

model_f = torch.load("apps/inference/models/forgery_model.pth", map_location="cpu")
qmodel_f = quantize_dynamic(model_f, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(qmodel_f, "apps/inference/models/forgery_model_q.pth")
print("Saved *_q.pth")
~~~
- Update inference env:
~~~ini
MODEL_ID_PATH=./models/id_model_q.pth
MODEL_F_PATH=./models/forgery_model_q.pth
MODEL_ID_FORMAT=torch
MODEL_F_FORMAT=torch
DEVICE=cpu
~~~

B) **ONNX → Dynamic INT8 with ONNX Runtime (portable, very fast on CPU)**  
- Export to ONNX (if not already), then quantize.
~~~python
# scripts/quantize_dynamic_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("apps/inference/models/id_model.onnx",
                 "apps/inference/models/id_model_int8.onnx",
                 weight_type=QuantType.QInt8)
quantize_dynamic("apps/inference/models/forgery_model.onnx",
                 "apps/inference/models/forgery_model_int8.onnx",
                 weight_type=QuantType.QInt8)
print("Saved *_int8.onnx")
~~~
- Update inference env:
~~~ini
MODEL_ID_PATH=./models/id_model_int8.onnx
MODEL_F_PATH=./models/forgery_model_int8.onnx
MODEL_ID_FORMAT=onnx
MODEL_F_FORMAT=onnx
DEVICE=cpu
~~~

C) **Static/Calibration-based INT8 (optional, best accuracy)**  
- Requires a small calibration set of real frames; more setup time. For hackathon timelines, dynamic INT8 is usually sufficient.

> **Tip:** After quantization, re-run a quick sanity notebook to check:  
> latency ↓, file size ↓, and AUC/threshold behavior stable (adjust `THRESH_*` if needed).

---

### 6) **Run Locally (Dev)**
~~~bash
# Start the inference API
# (ensure MODEL_* envs point to your chosen weights or quantized weights)
uvicorn main:app --host 0.0.0.0 --port ${FASTAPI_PORT:-8000}

# In a new terminal: start the web app
pnpm dev
~~~
Open http://localhost:3000 → **Sign up** → **Enroll** (short video or two photos) → **Face Login**.

---

### 7) **Deploy Inference on Railway (CPU plan)**
**Why Railway?** Simple build, free/low-cost tiers for demo, public HTTPS domain.

A) **Create service**
- Push your repo to GitHub.  
- In **Railway**, create a **New Project → Deploy from GitHub**.  
- Select the **inference** app or root repo (set working directory during build).  

B) **Build & Start Command**
- Railway detects Python; if not, set:
~~~text
Build:    pip install -r apps/inference/requirements.txt
Start:    uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
Root:     apps/inference
~~~
- Make sure your code reads `FASTAPI_PORT` but also accepts Railway’s `$PORT`. A simple approach:
  - Use `--port $PORT` in the Start command and ignore `FASTAPI_PORT` in production.

C) **Environment Variables (Railway → Variables)**
- Set all vars from **Section 3** for the **inference** service.  
- Point `MODEL_ID_PATH` and `MODEL_F_PATH` to the weight files you will provide at boot.

D) **Bring model weights at deploy time**
- Recommended: host weights on **GitHub Releases / Hugging Face / Google Drive** and **download on boot**.
~~~bash
# apps/inference/start.sh (make executable)
#!/usr/bin/env bash
set -e
mkdir -p models
# curl -L -o models/id_model_int8.onnx "https://<your-host>/id_model_int8.onnx"
# curl -L -o models/forgery_model_int8.onnx "https://<your-host>/forgery_model_int8.onnx"
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
~~~
- Change Railway **Start** to: `bash start.sh`

E) **CORS / Health**
- Ensure FastAPI enables CORS for your web origin.  
- Health endpoint `/healthz` should return **200**.  
- After deploy, Railway will provide a URL like `https://<service-name>.up.railway.app`.

---

### 8) **Deploy Web on Vercel (or any Node host)**
- Import the GitHub repo → Vercel.  
- Set web env vars (Section 3).  
- Set `INFERENCE_API_URL=https://<railway-subdomain>.up.railway.app`.  
- Build: `pnpm build`  •  Start: `pnpm start` (Vercel auto-detects Next.js).

---

### 9) **Production Tips**
- **Concurrency:** `--workers 2` (increase with CPU cores).  
- **Quantized ONNX** on CPU often gives the best cost/perf on Railway.  
- **Timeouts:** keep request size modest (8–24 frames) and compress JPEGs.  
- **Logging:** log only scores/outcomes; never raw images.  
- **CORS:** lock to your web origin.  
- **TLS:** Railway & Vercel provide HTTPS by default.

---

### 10) **Sanity Checklist**
- `/healthz` responds **200** on Railway.  
- Web envs point to Supabase + Railway inference URL.  
- **Enroll** returns `liveness_score` & `authenticity`.  
- **Login** returns exactly **one** message:
  - **“Fake image detected.”**
  - **“No match found for this account.”** *(1:1)* / **“No match found in the database.”** *(1:N)*
  - **“Login successful.”**
- UI shows **Match Score**, **Liveness Score**, **Authenticity Label**.  
- No raw frames persisted; embeddings are encrypted.

---

### 11) **Quick cURL Tests**
~~~bash
# Health
curl -s https://<railway-subdomain>.up.railway.app/healthz

# Login (example payload; replace with your base64 frames)
curl -s -X POST https://<railway-subdomain>.up.railway.app/verify/login \
  -H "Content-Type: application/json" \
  -d '{"user_id":"<uuid>","frames":["data:image/jpeg;base64,<...>","data:image/jpeg;base64,<...>"]}'
~~~

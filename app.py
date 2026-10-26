"""
brand-governance - ad policy checker web app (backend + frontend in one file).
Built by THE CIPHERERS (thecipherers@gmail.com).

Run:
    pip install flask google-genai pypdf pillow   (pypdf + pillow make it faster)
    (for a Claude key instead: pip install flask anthropic pypdf pillow)
    set GEMINI_API_KEY=your-key             (or: set ANTHROPIC_API_KEY=your-key)
    python app.py
Then open http://127.0.0.1:5000

The API key stays on your computer (server side). The browser never sees it.
"""

import base64
import hashlib
import io
import json
import mimetypes
import os
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB per request

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
# Used automatically if the main model is slow, busy or fails.
GEMINI_FALLBACK = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")
# Seconds to wait for ONE Gemini call before giving up on it.
GEMINI_TIMEOUT_MS = int(float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "45")) * 1000)
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
TEXT_TYPES = {".txt", ".md"}

SYSTEM_PROMPT = """You are an advertisement compliance reviewer.

You are given (1) a company's rules and policies and (2) an advertisement.
Your only job: decide whether the advertisement follows the policies, and if
it does not, tell the person exactly how to fix it.

Rules for your review:
- Judge ONLY against the policy document provided. Do not invent rules.
- Read everything in the ad: text, images, logos, colors, claims, fine print.
- If the ad breaks no rule, verdict is "PASS".
- If it breaks any rule, verdict is "FAIL" and you list EVERY broken rule.
- For each violation give three things, each short and specific:
    "rule":   the rule from the policy that is broken (name or quote it)
    "reason": what in the ad breaks it
    "fix":    a concrete change that would make the ad pass, such as the
              exact wording to remove or replace, or the element to add
- Always list in "checked" the short names (under 12 words each, at most 12
  items) of the most important policy rules you checked the ad against.
- If something is unclear or the policy doesn't cover it, do not fail the ad
  for it.

Respond with ONLY valid JSON in this exact shape:
{"verdict": "PASS" or "FAIL",
 "checked": ["<short rule name>", "..."],
 "violations": [{"rule": "...", "reason": "...", "fix": "..."}]}
For PASS, "violations" must be an empty list."""


# ----------------------------------------------------------------- helpers
class UserError(Exception):
    """An error whose message is safe and useful to show in the UI."""


def provider():
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return None


_pdf_cache = {}


def pdf_text(data):
    """Extract text from a PDF (needs the optional 'pypdf' package). Cached."""
    key = hashlib.sha256(data).hexdigest()
    if key in _pdf_cache:
        return _pdf_cache[key]
    text = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        text = ""
    _pdf_cache[key] = text
    return text


def shrink_image(data, mime):
    """Downscale very large images so uploads and AI reading are faster (needs optional 'pillow')."""
    try:
        from PIL import Image
    except ImportError:
        return data, mime
    try:
        im = Image.open(io.BytesIO(data))
        if max(im.size) <= 2000 and len(data) <= 3_000_000:
            return data, mime
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, (255, 255, 255))
            flat.paste(im, mask=im.split()[-1])
            im = flat
        else:
            im = im.convert("RGB")
        im.thumbnail((2000, 2000))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=88, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return data, mime


def read_upload(storage, policy=False):
    """Return ('text', str) or ('file', bytes, mime) for an uploaded file."""
    name = storage.filename or ""
    suffix = Path(name).suffix.lower()
    data = storage.read()
    if not data:
        raise UserError(f"{name} is empty.")
    if suffix == ".pdf":
        # A rules PDF with real text is much faster to send as plain text.
        if policy and os.environ.get("USE_PDF_TEXT", "1") != "0":
            text = pdf_text(data)
            if len(text) >= 200:
                return ("text", text[:150000])
        return ("file", data, "application/pdf")
    if suffix in IMAGE_TYPES:
        mime = mimetypes.guess_type(name)[0] or "image/png"
        data, mime = shrink_image(data, mime)
        return ("file", data, mime)
    if suffix in TEXT_TYPES:
        return ("text", data.decode("utf-8", errors="replace"))
    raise UserError(f"{name}: unsupported file type. Use PDF, PNG, JPG, GIF, WEBP, TXT or MD.")


def parse_result(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise UserError("The AI returned an answer in an unexpected format. Try again.")
    if not isinstance(result, dict):
        raise UserError("The AI returned an answer in an unexpected format. Try again.")
    verdict = str(result.get("verdict", "")).upper()
    if verdict not in ("PASS", "FAIL"):
        raise UserError("The AI returned an answer in an unexpected format. Try again.")
    violations = result.get("violations") or []
    clean = [
        {
            "rule": str(v.get("rule", "")).strip(),
            "reason": str(v.get("reason", "")).strip(),
            "fix": str(v.get("fix", "")).strip(),
        }
        for v in violations
        if isinstance(v, dict)
    ]
    checked = [str(c).strip() for c in (result.get("checked") or []) if str(c).strip()][:12]
    if verdict == "PASS":
        clean = []
    return {"verdict": verdict, "checked": checked, "violations": clean}


def _gemini_client():
    from google import genai
    from google.genai import types

    key = os.environ["GEMINI_API_KEY"]
    try:
        # attempts=1 turns off the library's default of 5 slow automatic retries.
        opts = types.HttpOptions(
            timeout=GEMINI_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    except Exception:
        try:
            opts = types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)
        except Exception:
            opts = None
    return genai.Client(api_key=key, http_options=opts) if opts else genai.Client(api_key=key)


def _gemini_call(client, model, contents, thinking):
    from google.genai import types

    cfg = {"system_instruction": SYSTEM_PROMPT, "response_mime_type": "application/json"}
    if thinking:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    response = client.models.generate_content(
        model=model, contents=contents, config=types.GenerateContentConfig(**cfg)
    )
    return response.text


def _is_config_error(e):
    """True when the model rejected our settings (e.g. a thinking level it lacks)."""
    if getattr(e, "code", None) == 400:
        return True
    return e.__class__.__name__ in ("ValidationError", "TypeError") or "thinking" in str(e).lower()


def run_gemini(policy, ad):
    from google.genai import types

    def part(item):
        if item[0] == "text":
            return item[1]
        return types.Part.from_bytes(data=item[1], mime_type=item[2])

    contents = [
        "COMPANY POLICY DOCUMENT:",
        part(policy),
        "ADVERTISEMENT TO REVIEW:",
        part(ad),
        "Does the advertisement follow the policy? Reply with the JSON only.",
    ]
    client = _gemini_client()
    models = [GEMINI_MODEL] + ([GEMINI_FALLBACK] if GEMINI_FALLBACK and GEMINI_FALLBACK != GEMINI_MODEL else [])
    first_error = None
    for model in models:
        # Least thinking first: it is the biggest speed-up. Fall back if the model rejects it.
        for thinking in ("minimal", "low", None):
            try:
                return _gemini_call(client, model, contents, thinking)
            except Exception as e:
                first_error = first_error or e
                if thinking is not None and _is_config_error(e):
                    continue  # try the next, gentler setting on the same model
                break  # slow, busy or broken: try the faster fallback model
    raise first_error


def run_claude(policy, ad):
    import anthropic

    def block(item):
        if item[0] == "text":
            return {"type": "text", "text": item[1]}
        data = base64.standard_b64encode(item[1]).decode()
        kind = "document" if item[2] == "application/pdf" else "image"
        return {"type": kind, "source": {"type": "base64", "media_type": item[2], "data": data}}

    client = anthropic.Anthropic(timeout=60.0, max_retries=1)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "COMPANY POLICY DOCUMENT:"},
                block(policy),
                {"type": "text", "text": "ADVERTISEMENT TO REVIEW:"},
                block(ad),
                {"type": "text", "text": "Does the advertisement follow the policy? Reply with the JSON only."},
            ],
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def run_check(policy, ad):
    which = provider()
    if which == "gemini":
        return parse_result(run_gemini(policy, ad))
    if which == "claude":
        return parse_result(run_claude(policy, ad))
    raise UserError(
        "No API key is set. Set GEMINI_API_KEY (free at aistudio.google.com) "
        "or ANTHROPIC_API_KEY, then restart the app."
    )


# ------------------------------------------------------------------ routes
@app.get("/api/status")
def status():
    which = provider()
    name = "brand-governance"
    by = {"gemini": "Google Gemini", "claude": "Claude"}.get(which, "AI")
    return jsonify({"provider": which, "name": name, "by": by})


@app.post("/api/check")
def check():
    try:
        ad_file = request.files.get("ad")
        if not ad_file or not ad_file.filename:
            raise UserError("Add the advertisement first.")

        policy_file = request.files.get("policy")
        policy_text = (request.form.get("policy_text") or "").strip()
        if policy_file and policy_file.filename:
            policy = read_upload(policy_file, policy=True)
        elif policy_text:
            policy = ("text", policy_text)
        else:
            raise UserError("Add the company rules first, as a file or as pasted text.")

        ad = read_upload(ad_file)
        started = time.time()
        result = run_check(policy, ad)
        result["seconds"] = round(time.time() - started, 1)
        print(f"[check] {provider()} answered in {result['seconds']}s", flush=True)
        return jsonify(result)
    except UserError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # provider/network errors
        return jsonify({"error": f"The check could not be completed: {str(e)[:300]}"}), 502


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Files are too large. Keep the total under 40 MB."}), 413


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/review")
def review():
    return Response(REVIEW_PAGE, mimetype="text/html")


# -------------------------------------------------------------- main page
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#F4F5FB">
<title>brand-governance</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='9' fill='%234A3FE0'/%3E%3Ccircle cx='15' cy='15' r='7' fill='none' stroke='white' stroke-width='2.5'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Syne:wght@600;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#F4F5FB; --surface:#FFFFFF; --ink:#111A36; --muted:#5A6482; --line:#DCE0EE;
  --brand:#4A3FE0; --brand-ink:#3B31C4; --brand-soft:#E9E7FD; --amber:#FFC247;
  --pass:#12805F; --pass-soft:#DFF4EC; --fail:#C7372B; --fail-soft:#FCE6E3; --focus:#1F6FEB;
  --display:"Syne",system-ui,sans-serif;
  --body:"Instrument Sans",system-ui,sans-serif;
  --shadow:0 1px 2px rgba(17,26,54,.06),0 14px 30px -18px rgba(74,63,224,.35);
}
*{box-sizing:border-box}
[hidden]{display:none!important}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--bg);color:var(--ink);font:400 17px/1.55 var(--body);-webkit-font-smoothing:antialiased}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px}
h1,h2,h3,h4{margin:0}
p{margin:0}
.ic{display:inline-flex;flex:none}
.ic svg{width:1.25em;height:1.25em;display:block;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

/* header + tabs */
.top{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;padding:18px 0}
.brand{display:flex;align-items:center;gap:10px;color:var(--ink);text-decoration:none;font:800 19px/1 var(--display)}
.brand svg{width:34px;height:34px;display:block}
.tabs{display:flex;gap:4px;background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:4px}
.tab{border:0;background:none;font:600 15px var(--body);color:var(--muted);padding:9px 18px;border-radius:999px;cursor:pointer}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{background:var(--ink);color:#fff}
@media (max-width:600px){.tabs{width:100%}.tab{flex:1;padding:10px 6px}}

/* hero */
.hero{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:32px;align-items:center;padding:28px 0 34px}
@media (max-width:860px){.hero{grid-template-columns:1fr;padding-top:12px}}
h1{font:800 clamp(32px,5.2vw,52px)/1.04 var(--display);letter-spacing:-.015em;max-width:15ch}
.lede{color:var(--muted);font-size:19px;max-width:44ch;margin-top:16px}
.hero-art{background:var(--brand-soft);border-radius:28px;padding:22px;max-width:380px;justify-self:end;width:100%}
@media (max-width:860px){.hero-art{justify-self:start;max-width:340px}}
.hero-art svg{display:block;width:100%;height:auto}
.scan{animation:scan 2.2s ease-in-out .4s 1 both}
@keyframes scan{from{transform:translateX(0)}to{transform:translateX(300px)}}
.badge{transform-box:fill-box;transform-origin:center;animation:pop .45s cubic-bezier(.2,.9,.3,1.3) both}
.badge.b1{animation-delay:1.2s}.badge.b2{animation-delay:1.9s}
@keyframes pop{from{transform:scale(0);opacity:0}to{transform:scale(1);opacity:1}}
@media (prefers-reduced-motion:reduce){.scan,.badge{animation:none}}

/* tool */
.banner{background:#FFF3CD;border:1px solid #E5CE7A;color:#5B4A00;padding:12px 16px;border-radius:10px;margin:0 0 20px}
.tool{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr);gap:24px;align-items:start;padding-bottom:56px}
@media (max-width:860px){.tool{grid-template-columns:1fr}}
.stack{display:grid;gap:18px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow)}
.panel h2{font:800 21px/1.2 var(--display)}
.hint{color:var(--muted);font-size:15px;margin:4px 0 14px}
.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;text-align:center;
  min-height:118px;padding:18px;border:2px dashed #BAC1D8;border-radius:12px;cursor:pointer;background:#FAFBFE;
  transition:border-color .15s,background .15s}
.drop .ic{color:var(--brand);font-size:26px;margin-bottom:2px}
.drop:hover,.drop.over{border-color:var(--brand);background:var(--brand-soft)}
.drop:focus-within{outline:3px solid var(--focus);outline-offset:2px}
.drop.has-file{border-style:solid;border-color:var(--brand);background:var(--brand-soft)}
.drop strong{font-weight:700;word-break:break-word}
.drop small{color:var(--muted)}
.vh{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.sep{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:15px;margin:14px 0 10px}
.sep::before,.sep::after{content:"";flex:1;border-top:1px solid var(--line)}
textarea{width:100%;min-height:92px;resize:vertical;padding:12px;border:1px solid var(--line);border-radius:10px;
  font:400 16px/1.45 var(--body);color:var(--ink);background:#fff}
.link{background:none;border:0;color:var(--brand);text-decoration:underline;cursor:pointer;font:inherit;padding:0;margin-top:10px}
.stage{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#fff;margin-top:14px}
.stage img{display:block;max-width:100%;max-height:300px;margin:0 auto}
.pdfcard{display:flex;align-items:center;gap:12px;padding:18px}
.pdfcard .ic{font-size:30px;color:var(--brand)}
.pdfcard span{display:block;color:var(--muted);font-size:14px}
.stage.scanning::after{content:"";position:absolute;top:0;bottom:0;left:-30%;width:30%;
  background:linear-gradient(90deg,transparent,rgba(74,63,224,.3));border-right:2px solid var(--brand);animation:sweep 1.3s ease-in-out infinite}
@keyframes sweep{to{left:100%}}
@media (prefers-reduced-motion:reduce){.stage.scanning::after{left:0;width:100%;opacity:.16;animation:none;border:0}}

.btn{display:inline-flex;align-items:center;justify-content:center;gap:9px;background:var(--brand);color:#fff;border:2px solid var(--brand);
  border-radius:12px;padding:14px 22px;font:700 17px var(--body);cursor:pointer;text-decoration:none;transition:background .15s}
.btn:hover{background:var(--brand-ink);border-color:var(--brand-ink)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn.block{width:100%;padding:16px 22px}
.btn.ghost{background:#fff;color:var(--ink);border-color:var(--line)}
.btn.ghost:hover{border-color:var(--ink);background:#fff}
.btn.green{background:var(--pass);border-color:var(--pass)}
.btn.green:hover{background:#0E6A50;border-color:#0E6A50}

/* verdict */
aside{position:sticky;top:20px}
@media (max-width:860px){aside{position:static}}
.sheet{background:var(--surface);border:1px solid var(--line);border-top:6px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow);min-height:320px}
.sheet.pass{border-top-color:var(--pass)}.sheet.fail{border-top-color:var(--fail)}
.sheet>h2{font:800 21px/1.2 var(--display);margin-bottom:6px}
.empty{display:grid;justify-items:center;gap:10px;text-align:center;color:var(--muted);padding:26px 8px}
.empty svg{width:120px;height:auto}
.working{display:flex;align-items:center;gap:12px;color:var(--muted);padding:18px 0}
.spin{width:20px;height:20px;border-radius:50%;border:3px solid var(--line);border-top-color:var(--brand);animation:rot .8s linear infinite;flex:none}
@keyframes rot{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.spin{animation:none}}
.err{color:var(--fail);font-weight:600;padding:14px 0}
.stampwrap{padding:22px 0 18px}
.stamp{display:inline-block;border:5px double currentColor;border-radius:8px;padding:6px 24px;
  font:800 clamp(40px,8vw,60px)/1 var(--display);letter-spacing:.05em;transform:rotate(-6deg)}
.stamp.pass{color:var(--pass)}.stamp.fail{color:var(--fail)}
.stamp.slam{animation:slam .38s cubic-bezier(.2,.9,.3,1.2) both}
@keyframes slam{0%{transform:rotate(-15deg) scale(1.8);opacity:0}55%{opacity:1}100%{transform:rotate(-6deg) scale(1);opacity:1}}
@media (prefers-reduced-motion:reduce){.stamp.slam{animation:none}}
.vtitle{font:800 22px/1.2 var(--display)}
.vsub{color:var(--muted);margin-top:6px}
.vlist{list-style:none;margin:18px 0 0;padding:0;display:grid;gap:14px}
.vcard{border:1px solid var(--line);border-left:5px solid var(--fail);border-radius:12px;padding:16px;background:#fff}
.vcard h4{font:700 17px/1.3 var(--body)}
.vcard .wrong{margin-top:6px;color:var(--muted)}
.fixbox{display:flex;gap:10px;margin-top:12px;padding:12px 14px;border-radius:10px;background:var(--brand-soft)}
.fixbox .ic{color:var(--brand);margin-top:2px}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}
.checked{list-style:none;margin:16px 0 0;padding:0;display:grid;gap:8px}
.checked li{display:flex;gap:10px;align-items:flex-start}
.checked .ic{color:var(--pass);margin-top:2px}
.human{margin-top:22px;padding:18px;border-radius:14px;background:var(--pass-soft);display:grid;gap:12px}
.human .row{display:flex;gap:12px;align-items:flex-start}
.human .row .ic{color:var(--pass);font-size:24px}
.human p{color:#124B3A}
.fine{color:var(--muted);font-size:14px;margin-top:14px}

/* how it works */
.page{padding:20px 0 64px}
.page h1{max-width:none;font-size:clamp(30px,4.6vw,44px)}
.steps{list-style:none;counter-reset:s;margin:30px 0 0;padding:0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}
@media (max-width:860px){.steps{grid-template-columns:1fr}}
.steps li{counter-increment:s;background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow)}
.steps li::before{content:counter(s);display:grid;place-items:center;width:38px;height:38px;border-radius:50%;background:var(--ink);color:#fff;font:800 18px var(--display);margin-bottom:14px}
.steps h3{font:800 19px/1.2 var(--display);margin-bottom:6px}
.steps p,.outcome p{color:var(--muted)}
.outcomes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:18px}
@media (max-width:860px){.outcomes{grid-template-columns:1fr}}
.outcome{border-radius:18px;padding:24px;display:grid;gap:10px}
.outcome h3{font:800 20px/1.2 var(--display);display:flex;align-items:center;gap:10px}
.outcome.f{background:var(--fail-soft)}.outcome.f h3{color:var(--fail)}
.outcome.p{background:var(--pass-soft)}.outcome.p h3{color:var(--pass)}
.outcome p{color:#33405E}
.know{margin-top:34px}
.know h2{font:800 22px/1.2 var(--display);margin-bottom:10px}
.know ul{margin:0;padding-left:20px;display:grid;gap:8px;max-width:70ch;color:#33405E}

/* about */
.about{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr);gap:40px;align-items:center;padding-top:10px}
@media (max-width:860px){.about{grid-template-columns:1fr}}
.wheelbtn{background:none;border:0;padding:0;cursor:pointer;display:block;width:100%;max-width:360px;border-radius:50%}
.wheelbtn svg{display:block;width:100%;height:auto}
.wheel text{font-family:var(--display);font-weight:600}
#inner{transition:transform .6s cubic-bezier(.3,.8,.3,1)}
@media (prefers-reduced-motion:reduce){#inner{transition:none}}
.cipher-line{margin-top:14px;color:var(--muted);max-width:360px}
.cipher-line strong{color:var(--ink);font:800 18px var(--display);letter-spacing:.04em;display:block;margin-top:2px}
.team{font:800 clamp(34px,6vw,60px)/1 var(--display);letter-spacing:-.01em}
.about p.desc{color:var(--muted);font-size:19px;margin-top:16px;max-width:46ch}
.contact{margin-top:26px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:18px 20px;box-shadow:var(--shadow)}
.contact .ic{font-size:26px;color:var(--brand)}
.contact a{color:var(--ink);font-weight:700;font-size:18px;word-break:break-all}
.contact .who{flex:1;min-width:200px}
.contact small{display:block;color:var(--muted)}
footer{border-top:1px solid var(--line);padding:22px 0 40px;color:var(--muted);font-size:15px}
footer a{color:var(--ink)}

/* ===== visual upgrade ===== */
body{position:relative;overflow-x:hidden;background:#EDEFFB}
.wrap{position:relative;z-index:1}
.bg{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
.bg .orb{position:absolute;border-radius:50%;filter:blur(72px);opacity:.5;animation:orb 22s ease-in-out infinite alternate}
.o1{width:560px;height:560px;left:-140px;top:-130px;background:#7C6CFF}
.o2{width:480px;height:480px;right:-130px;top:28%;background:#FFC247;animation-duration:27s}
.o3{width:540px;height:540px;left:28%;bottom:-240px;background:#5EE0B5;animation-duration:31s;opacity:.4}
@keyframes orb{to{transform:translate(70px,50px) scale(1.12)}}
.gridlines{position:absolute;inset:0;background-image:linear-gradient(rgba(74,63,224,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(74,63,224,.08) 1px,transparent 1px);background-size:56px 56px;-webkit-mask-image:radial-gradient(ellipse at 50% 28%,#000 15%,transparent 78%);mask-image:radial-gradient(ellipse at 50% 28%,#000 15%,transparent 78%)}
#net{position:absolute;inset:0;width:100%;height:100%}
#floaters{position:absolute;inset:0}
.fl{position:absolute;animation:drift var(--d,16s) ease-in-out infinite alternate}
.fl svg{display:block;width:100%;height:auto;filter:drop-shadow(0 10px 14px rgba(17,26,54,.18))}
@keyframes drift{from{transform:translate(0,0) rotate(var(--r0,0deg))}to{transform:translate(var(--dx,30px),var(--dy,-40px)) rotate(var(--r1,10deg))}}
#fx{position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:60}

/* header + glass */
.brand{font-size:26px;gap:12px}
body[data-tab="check"] #appname{display:none}
.brand svg{width:46px;height:46px}
.tabs,.panel,.sheet,.steps li,.contact{background:rgba(255,255,255,.88);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px)}
.lede{color:#3B4668}

/* giant wordmark */
.titlebar{padding:8px 0 0}
h1.wordmark{font:800 clamp(40px,6.4vw,74px)/1.02 var(--display);letter-spacing:-.025em;max-width:none;margin:0;text-shadow:0 12px 44px rgba(74,63,224,.3)}
@media (max-width:760px){h1.wordmark{font-size:clamp(30px,9vw,64px)}}
.wordmark .wd{display:inline-block;white-space:nowrap}
.wordmark .ch{display:inline-block;animation:drop .7s cubic-bezier(.2,.9,.3,1.25) backwards,wave .9s ease 1.4s 1;animation-delay:calc(var(--i)*45ms),calc(1.4s + var(--i)*55ms);transition:transform .2s,color .2s;cursor:default}
.wordmark .ch:hover{transform:translateY(-10px) rotate(-5deg);color:var(--brand)}
.wordmark .hy{color:var(--brand)}
@keyframes drop{from{transform:translateY(-70%) rotate(-10deg);opacity:0}to{transform:none;opacity:1}}
@keyframes wave{0%,100%{color:var(--ink)}50%{color:var(--brand)}}
.uline{height:9px;width:min(280px,46%);margin-top:10px;border-radius:99px;background:linear-gradient(90deg,var(--brand),var(--amber));transform-origin:left;animation:draw .8s ease 1.7s backwards}
@keyframes draw{from{transform:scaleX(0)}to{transform:scaleX(1)}}

/* hero */
.hero{padding:22px 0 30px}
.tagline{font:800 clamp(24px,3.6vw,40px)/1.1 var(--display);letter-spacing:-.01em;max-width:18ch}
.feats{list-style:none;margin:24px 0 0;padding:0;display:grid;gap:10px;max-width:440px}
.feat{display:flex;align-items:center;gap:12px;background:rgba(255,255,255,.86);border:1px solid var(--line);border-radius:14px;padding:10px 14px;font-weight:600;-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px)}
.fi{display:grid;place-items:center;width:38px;height:38px;border-radius:50%;background:var(--brand-soft);color:var(--brand);flex:none}
.fi.amber{background:#FFF0CC;color:#9A6A00}.fi.green{background:var(--pass-soft);color:var(--pass)}
.fi .ic svg{width:21px;height:21px}
.hero-art{max-width:460px;background:rgba(233,231,253,.85);box-shadow:var(--shadow)}
@keyframes scan{from{transform:translateX(0)}to{transform:translateX(340px)}}
.lens{animation:lens 8s ease-in-out infinite alternate}
@keyframes lens{from{transform:translate(-56px,-26px)}to{transform:translate(52px,62px)}}
.spark{transform-box:fill-box;transform-origin:center;animation:tw 2.6s ease-in-out infinite}
.sp2{animation-delay:-.9s}.sp3{animation-delay:-1.7s}
@keyframes tw{50%{transform:scale(.35);opacity:.35}}

/* panels */
.panel h2{display:flex;align-items:center;gap:10px}
.hico{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:var(--brand-soft);color:var(--brand);flex:none}
.hico.amber{background:#FFF0CC;color:#9A6A00}
.hico .ic svg{width:22px;height:22px}
.btn.block{box-shadow:0 14px 30px -12px rgba(74,63,224,.7)}
.btn.block:disabled{box-shadow:none}
.sheet.fail{animation:shake .45s}
.sheet.pass{animation:glow 1.2s ease-out}
@keyframes shake{20%{transform:translateX(-6px)}40%{transform:translateX(6px)}60%{transform:translateX(-4px)}80%{transform:translateX(3px)}}
@keyframes glow{from{box-shadow:0 0 0 0 rgba(18,128,95,.5)}to{box-shadow:0 0 0 26px rgba(18,128,95,0)}}
@media (prefers-reduced-motion:reduce){
  .bg .orb,.fl,.lens,.spark,.wordmark .ch,.uline,.sheet.fail,.sheet.pass{animation:none}
}
</style>
</head>
<body>
<div class="bg" aria-hidden="true">
  <div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
  <div class="gridlines"></div>
  <canvas id="net"></canvas>
  <div id="floaters"></div>
</div>
<canvas id="fx" aria-hidden="true"></canvas>
<div class="wrap">
  <header class="top">
    <a class="brand" href="#check" aria-label="brand-governance home">
      <svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="#4A3FE0"/><circle cx="15" cy="15" r="7" fill="none" stroke="#fff" stroke-width="2.5"/><path d="M11.8 15.2l2.4 2.4 4.2-4.6" fill="none" stroke="#FFC247" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M20.5 20.5L25 25" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/></svg>
      <span id="appname">brand-governance</span>
    </a>
    <nav class="tabs" role="tablist" aria-label="Sections">
      <button class="tab" role="tab" id="tab-check" aria-controls="panel-check" aria-selected="true">Check</button>
      <button class="tab" role="tab" id="tab-how" aria-controls="panel-how" aria-selected="false" tabindex="-1">How it works</button>
      <button class="tab" role="tab" id="tab-about" aria-controls="panel-about" aria-selected="false" tabindex="-1">About</button>
    </nav>
  </header>

  <main>
  <!-- CHECK -->
  <section id="panel-check" role="tabpanel" aria-labelledby="tab-check">
    <div class="titlebar">
      <h1 class="wordmark" id="wordmark">brand-governance</h1>
      <div class="uline" aria-hidden="true"></div>
    </div>
    <div class="hero">
      <div>
        <h2 class="tagline">Does your ad follow the rules?</h2>
        <p class="lede">Add your company's rules and your ad. You get a pass, or clear steps to fix what breaks.</p>
        <ul class="feats">
          <li class="feat"><span class="fi"><span class="ic" data-icon="eye" aria-hidden="true"></span></span>Reads PDFs and images</li>
          <li class="feat"><span class="fi amber"><span class="ic" data-icon="edit" aria-hidden="true"></span></span>Fix steps for every broken rule</li>
          <li class="feat"><span class="fi green"><span class="ic" data-icon="person" aria-hidden="true"></span></span>Human sign-off when it passes</li>
        </ul>
      </div>
      <div class="hero-art">
        <svg viewBox="0 0 400 330" role="img" aria-label="An ad being scanned by a magnifying lens, with a green check for what passes and a red flag for what breaks">
          <defs>
            <clipPath id="adclip"><rect x="40" y="34" width="240" height="250" rx="18"/></clipPath>
            <linearGradient id="scanG" x1="0" x2="1"><stop offset="0" stop-color="#4A3FE0" stop-opacity="0"/><stop offset="1" stop-color="#4A3FE0" stop-opacity=".3"/></linearGradient>
          </defs>
          <rect x="58" y="54" width="240" height="250" rx="18" fill="#C9C3F7"/>
          <rect x="40" y="34" width="240" height="250" rx="18" fill="#fff" stroke="#DCE0EE"/>
          <g clip-path="url(#adclip)">
            <rect x="40" y="34" width="240" height="116" fill="#111A36"/>
            <circle cx="228" cy="88" r="34" fill="#4A3FE0"/>
            <circle cx="252" cy="114" r="20" fill="#FFC247"/>
            <rect x="62" y="62" width="104" height="12" rx="6" fill="#fff"/>
            <rect x="62" y="84" width="72" height="12" rx="6" fill="#fff" opacity=".55"/>
            <rect x="62" y="172" width="168" height="10" rx="5" fill="#C9CEE0"/>
            <rect x="62" y="192" width="136" height="10" rx="5" fill="#DCE0EE"/>
            <rect x="62" y="212" width="100" height="10" rx="5" fill="#DCE0EE"/>
            <rect x="62" y="240" width="84" height="28" rx="14" fill="#4A3FE0"/>
            <g class="scan"><rect x="-48" y="34" width="48" height="250" fill="url(#scanG)"/><rect x="-2" y="34" width="2.5" height="250" fill="#4A3FE0"/></g>
          </g>
          <g class="lens">
            <circle cx="150" cy="150" r="46" fill="rgba(255,255,255,.3)" stroke="#4A3FE0" stroke-width="7"/>
            <path d="M183 183l32 32" stroke="#4A3FE0" stroke-width="11" stroke-linecap="round"/>
            <path d="M133 152l12 12 24-26" fill="none" stroke="#FFC247" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
          </g>
          <g class="badge b1"><circle cx="312" cy="96" r="28" fill="#12805F" stroke="#E9E7FD" stroke-width="5"/><path d="M300 96.5l8 8L324 87" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></g>
          <g class="badge b2"><circle cx="312" cy="214" r="28" fill="#C7372B" stroke="#E9E7FD" stroke-width="5"/><path d="M312 200v16" stroke="#fff" stroke-width="5" stroke-linecap="round"/><circle cx="312" cy="226" r="3.2" fill="#fff"/></g>
          <g transform="translate(352 40)"><path class="spark" d="M0-13L3.5-3.5 13 0 3.5 3.5 0 13-3.5 3.5-13 0-3.5-3.5Z" fill="#FFC247"/></g>
          <g transform="translate(24 300)"><path class="spark sp2" d="M0-11L3-3 11 0 3 3 0 11-3 3-11 0-3-3Z" fill="#4A3FE0"/></g>
          <g transform="translate(360 292)"><path class="spark sp3" d="M0-9L2.5-2.5 9 0 2.5 2.5 0 9-2.5 2.5-9 0-2.5-2.5Z" fill="#12805F"/></g>
        </svg>
      </div>
    </div>

    <div class="banner" id="banner" role="alert" hidden></div>

    <div class="tool">
      <div class="stack">
        <section class="panel">
          <h2><span class="hico"><span class="ic" data-icon="shield" aria-hidden="true"></span></span>Company rules</h2>
          <p class="hint">The policy your ad has to follow.</p>
          <label class="drop" id="policyDrop">
            <input class="vh" type="file" id="policyFile" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.txt,.md">
            <span class="ic" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 16V5M7.5 9.5L12 5l4.5 4.5"/><path d="M5 19h14"/></svg></span>
            <strong id="policyName">Drop the rules file here, or click to choose</strong>
            <small>PDF, image, TXT or MD</small>
          </label>
          <button class="link" id="policyClear" hidden>Remove file</button>
          <div class="sep">or paste the rules</div>
          <textarea id="policyText" placeholder="For example: No claims of guaranteed results. The company logo must appear." aria-label="Paste company rules"></textarea>
        </section>

        <section class="panel">
          <h2><span class="hico amber"><span class="ic" data-icon="image" aria-hidden="true"></span></span>Advertisement</h2>
          <p class="hint">The ad you want checked.</p>
          <label class="drop" id="adDrop">
            <input class="vh" type="file" id="adFile" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp">
            <span class="ic" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 16V5M7.5 9.5L12 5l4.5 4.5"/><path d="M5 19h14"/></svg></span>
            <strong id="adName">Drop the ad here, or click to choose</strong>
            <small>PDF, PNG, JPG, GIF or WEBP</small>
          </label>
          <button class="link" id="adClear" hidden>Remove file</button>
          <div class="stage" id="stage" hidden></div>
        </section>

        <button class="btn block" id="go" disabled>Check ad</button>
      </div>

      <aside>
        <section class="sheet" id="sheet" aria-labelledby="vh" aria-live="polite">
          <h2 id="vh">Verdict</h2>
          <div id="result"></div>
        </section>
      </aside>
    </div>
  </section>

  <!-- HOW IT WORKS -->
  <section id="panel-how" role="tabpanel" aria-labelledby="tab-how" class="page" hidden>
    <h1>How it works</h1>
    <p class="lede">Three steps, then you know where your ad stands.</p>
    <ol class="steps">
      <li><h3>Add the rules</h3><p>Upload your company's policy, or paste the rules into the box.</p></li>
      <li><h3>Add the ad</h3><p>Upload a PDF or an image. The checker reads the text, logos, claims and fine print.</p></li>
      <li><h3>Read the verdict</h3><p>You get a pass, or a list of what to change.</p></li>
    </ol>
    <div class="outcomes">
      <div class="outcome f">
        <h3><span class="ic" aria-hidden="true"></span>If the ad fails</h3>
        <p>Each broken rule comes with what is wrong and how to fix it. Change the ad, upload the new version and check again until it passes.</p>
      </div>
      <div class="outcome p">
        <h3><span class="ic" aria-hidden="true"></span>If the ad passes</h3>
        <p>You can open a checklist in a new tab for a person to review and sign off. Print it, save it as a PDF or email it.</p>
      </div>
    </div>
    <div class="know">
      <h2>Good to know</h2>
      <ul>
        <li>The checker only judges the ad against the rules you give it. It adds no rules of its own.</li>
        <li>An AI can miss things. That is why a pass comes with a checklist for a person to confirm.</li>
        <li>Your files are sent to the AI service connected to this app to be read. Don't upload anything you are not allowed to share.</li>
      </ul>
    </div>
  </section>

  <!-- ABOUT -->
  <section id="panel-about" role="tabpanel" aria-labelledby="tab-about" class="page" hidden>
    <div class="about">
      <div>
        <button class="wheelbtn" id="wheelbtn" aria-label="Turn the cipher wheel by one letter">
          <svg class="wheel" viewBox="0 0 320 320" aria-hidden="true">
            <circle cx="160" cy="160" r="154" fill="#111A36"/>
            <g id="outer"></g>
            <g id="inner" style="transform-origin:160px 160px">
              <circle cx="160" cy="160" r="108" fill="#4A3FE0"/>
              <g id="innerLetters"></g>
            </g>
            <circle cx="160" cy="160" r="52" fill="#FFC247"/>
            <text x="160" y="154" text-anchor="middle" font-size="8.5" fill="#111A36" style="font-weight:800">THE</text>
            <text x="160" y="167" text-anchor="middle" font-size="8.5" fill="#111A36" style="font-weight:800">CIPHERERS</text>
            <path d="M151 4h18l-9 18z" fill="#FFC247"/>
          </svg>
        </button>
        <p class="cipher-line">Turn the wheel. THE CIPHERERS becomes <strong id="enc" aria-live="polite">WKH FLSKHUHUV</strong></p>
      </div>
      <div>
        <h1 class="team">THE CIPHERERS</h1>
        <p class="desc">The team behind brand-governance. We built it so you can find out fast whether an ad follows your company's rules, and know exactly what to change when it doesn't.</p>
        <div class="contact">
          <span class="ic" aria-hidden="true"></span>
          <div class="who"><small>Contact us</small><a href="mailto:thecipherers@gmail.com">thecipherers@gmail.com</a></div>
          <button class="btn ghost" id="copyMail" type="button">Copy email</button>
        </div>
      </div>
    </div>
  </section>
  </main>

  <footer>Built by <a href="#about">THE CIPHERERS</a>. Contact: <a href="mailto:thecipherers@gmail.com">thecipherers@gmail.com</a></footer>
</div>

<script>
(function(){
  "use strict";
  var $=function(id){return document.getElementById(id)};
  function h(tag,props,kids){
    var n=document.createElement(tag);
    if(props){Object.keys(props).forEach(function(k){
      var v=props[k];
      if(k==="class") n.className=v;
      else if(k==="text") n.textContent=v;
      else if(k.slice(0,2)==="on") n.addEventListener(k.slice(2),v);
      else n.setAttribute(k,v);
    });}
    (kids||[]).forEach(function(c){ if(c==null) return; n.appendChild(typeof c==="string"?document.createTextNode(c):c); });
    return n;
  }
  var P={
    check:'<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    warn:'<path d="M12 8v5M12 16.5v.01"/><path d="M10.3 3.9L2.5 17.5a2 2 0 001.7 3h15.6a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/>',
    edit:'<path d="M4 20h4L19 9a2.8 2.8 0 00-4-4L4 16v4z"/><path d="M13.5 6.5l4 4"/>',
    file:'<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/>',
    external:'<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5"/>',
    copy:'<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 012-2h9"/>',
    mail:'<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3.5 7l8.5 6.5L20.5 7"/>',
    person:'<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0113 0"/><path d="M16 11.5l2 2 3.5-4"/>',
    upload:'<path d="M12 16V5M7.5 9.5L12 5l4.5 4.5"/><path d="M5 19h14"/>'
  };
  function icon(name){
    var s=document.createElement("span"); s.className="ic"; s.setAttribute("aria-hidden","true");
    s.innerHTML='<svg viewBox="0 0 24 24">'+P[name]+'</svg>'; return s;
  }
  function setIcon(el,name){ el.innerHTML='<svg viewBox="0 0 24 24">'+P[name]+'</svg>'; }

  /* static icons */
  setIcon(document.querySelector(".outcome.f .ic"),"warn");
  setIcon(document.querySelector(".outcome.p .ic"),"check");
  setIcon(document.querySelector(".contact .ic"),"mail");

  /* ---------- tabs ---------- */
  var TABS=["check","how","about"];
  function apply(){
    var name=(location.hash||"#check").slice(1);
    if(TABS.indexOf(name)<0) name="check";
    TABS.forEach(function(t){
      var on=t===name;
      $("tab-"+t).setAttribute("aria-selected",on?"true":"false");
      $("tab-"+t).tabIndex=on?0:-1;
      $("panel-"+t).hidden=!on;
    });
    document.body.setAttribute("data-tab",name);
  }
  TABS.forEach(function(t,i){
    var b=$("tab-"+t);
    b.addEventListener("click",function(){ location.hash="#"+t; window.scrollTo(0,0); });
    b.addEventListener("keydown",function(e){
      var d=e.key==="ArrowRight"?1:e.key==="ArrowLeft"?-1:0; if(!d) return;
      e.preventDefault();
      var nt=TABS[(i+d+TABS.length)%TABS.length]; location.hash="#"+nt; $("tab-"+nt).focus();
    });
  });
  window.addEventListener("hashchange",function(){apply(); window.scrollTo(0,0);});
  apply();

  /* ---------- status ---------- */
  var state={provider:null,by:"AI"};
  fetch("/api/status").then(function(r){return r.json()}).then(function(s){
    state.provider=s.provider; state.by=s.by;
    $("appname").textContent=s.name; document.title=s.name;
    if(!s.provider){
      var b=$("banner"); b.hidden=false;
      b.textContent="No API key is set, so checks will fail. Set GEMINI_API_KEY (free at aistudio.google.com) or ANTHROPIC_API_KEY, then restart the app.";
    }
  }).catch(function(){});

  /* ---------- inputs ---------- */
  var policyFile=null, adFile=null, previewUrl=null;
  var go=$("go"), result=$("result"), sheet=$("sheet"), stage=$("stage");

  function refresh(){
    var hasPolicy=!!policyFile || $("policyText").value.trim().length>0;
    go.disabled=!(hasPolicy && adFile);
  }
  function emptyState(){
    sheet.className="sheet";
    var svg='<svg viewBox="0 0 120 90" aria-hidden="true"><rect x="22" y="8" width="62" height="76" rx="8" fill="#fff" stroke="#BAC1D8" stroke-width="2" stroke-dasharray="5 4"/><rect x="32" y="22" width="30" height="5" rx="2.5" fill="#DCE0EE"/><rect x="32" y="34" width="42" height="5" rx="2.5" fill="#DCE0EE"/><rect x="32" y="46" width="24" height="5" rx="2.5" fill="#DCE0EE"/><circle cx="82" cy="58" r="17" fill="#E9E7FD" stroke="#4A3FE0" stroke-width="3"/><path d="M75 58.5l5 5 9-10" fill="none" stroke="#4A3FE0" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    var d=h("div",{class:"empty"},[h("p",{text:"Add the rules and your ad, then press Check ad. Your result appears here."})]);
    d.insertAdjacentHTML("afterbegin",svg);
    show(d);
  }
  function show(node){ result.textContent=""; result.appendChild(node); }

  function bindDrop(dropId,inputId,onPick){
    var drop=$(dropId), input=$(inputId);
    /* clear on click so re-choosing a file with the same name still counts as a new upload */
    input.addEventListener("click",function(){ input.value=""; });
    input.addEventListener("change",function(){ if(input.files[0]) onPick(input.files[0]); });
    ["dragenter","dragover"].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();drop.classList.add("over")})});
    ["dragleave","drop"].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();drop.classList.remove("over")})});
    drop.addEventListener("drop",function(e){ if(e.dataTransfer.files[0]) onPick(e.dataTransfer.files[0]); });
  }
  bindDrop("policyDrop","policyFile",function(f){
    policyFile=f; $("policyName").textContent=f.name; $("policyDrop").classList.add("has-file");
    $("policyClear").hidden=false; emptyState(); refresh();
  });
  $("policyClear").addEventListener("click",function(){
    policyFile=null; $("policyFile").value=""; $("policyName").textContent="Drop the rules file here, or click to choose";
    $("policyDrop").classList.remove("has-file"); $("policyClear").hidden=true; refresh();
  });
  bindDrop("adDrop","adFile",function(f){
    adFile=f; $("adName").textContent=f.name; $("adDrop").classList.add("has-file"); $("adClear").hidden=false;
    stage.hidden=false; stage.textContent="";
    if(previewUrl){URL.revokeObjectURL(previewUrl); previewUrl=null;}
    if(f.type.indexOf("image/")===0){
      previewUrl=URL.createObjectURL(f);
      stage.appendChild(h("img",{src:previewUrl,alt:"Preview of the ad"}));
    }else{
      var c=h("div",{class:"pdfcard"},[icon("file"),h("div",{},[h("strong",{text:f.name}),h("span",{text:"PDF document"})])]);
      stage.appendChild(c);
    }
    emptyState(); refresh();
  });
  $("adClear").addEventListener("click",function(){
    adFile=null; $("adFile").value=""; $("adName").textContent="Drop the ad here, or click to choose";
    $("adDrop").classList.remove("has-file"); $("adClear").hidden=true; stage.hidden=true; stage.textContent=""; refresh();
  });
  $("policyText").addEventListener("input",refresh);

  /* ---------- check ---------- */
  function stamp(v){
    return h("div",{class:"stampwrap"},[h("span",{class:"stamp "+(v==="PASS"?"pass":"fail")+" slam",text:v})]);
  }

  function fixText(d){
    var lines=["Ad: "+adFile.name,"Result: FAIL",""];
    d.violations.forEach(function(v,i){
      lines.push((i+1)+". "+v.rule);
      lines.push("   Problem: "+v.reason);
      lines.push("   Fix: "+(v.fix||"Change the ad so it no longer breaks this rule."));
      lines.push("");
    });
    return lines.join("\n");
  }
  function copy(text,btn,done){
    var orig=btn.textContent;
    function ok(){ btn.textContent=done; setTimeout(function(){btn.textContent=orig;},1800); }
    function fallback(){
      var t=h("textarea",{style:"position:fixed;opacity:0"}); t.value=text; document.body.appendChild(t); t.select();
      try{document.execCommand("copy"); ok();}catch(e){btn.textContent="Copy failed";} document.body.removeChild(t);
    }
    if(navigator.clipboard&&navigator.clipboard.writeText){ navigator.clipboard.writeText(text).then(ok,fallback); } else fallback();
  }

  function took(d){ return d.seconds?h("p",{class:"fine",text:"Checked in "+d.seconds+" seconds."}):null; }

  function renderFail(d){
    sheet.className="sheet fail";
    var box=h("div"), n=d.violations.length;
    box.appendChild(stamp("FAIL"));
    box.appendChild(h("h3",{class:"vtitle",text:n?("Fix "+n+(n===1?" thing":" things")+", then check again"):"This ad breaks the company's rules"}));
    box.appendChild(h("p",{class:"vsub",text:"Each card shows the rule, what is wrong in your ad, and how to fix it."}));
    var ul=h("ul",{class:"vlist"});
    d.violations.forEach(function(v){
      var fix=v.fix||"Change the ad so it no longer breaks this rule, then check again.";
      ul.appendChild(h("li",{class:"vcard"},[
        h("h4",{text:v.rule||"Rule not named"}),
        h("p",{class:"wrong",text:v.reason}),
        h("div",{class:"fixbox"},[icon("edit"),h("div",{},[h("strong",{text:"How to fix: "}),fix])])
      ]));
    });
    box.appendChild(ul);
    var copyBtn=h("button",{class:"btn ghost",type:"button",text:"Copy fix list"});
    copyBtn.addEventListener("click",function(){copy(fixText(d),copyBtn,"Copied");});
    var again=h("button",{class:"btn",type:"button"},[icon("upload"),"Upload the fixed ad"]);
    again.addEventListener("click",function(){ $("adFile").click(); });
    box.appendChild(h("div",{class:"actions"},[again,copyBtn]));
    var t1=took(d); if(t1) box.appendChild(t1);
    show(box);
  }

  function renderPass(d){
    sheet.className="sheet pass";
    var box=h("div");
    box.appendChild(stamp("PASS"));
    box.appendChild(h("h3",{class:"vtitle",text:"This ad follows the company's rules"}));
    if(d.checked&&d.checked.length){
      box.appendChild(h("p",{class:"vsub",text:"Rules the checker looked at:"}));
      var ul=h("ul",{class:"checked"});
      d.checked.forEach(function(r){ ul.appendChild(h("li",{},[icon("check"),h("span",{text:r})])); });
      box.appendChild(ul);
    }
    var payload={ad:adFile.name,at:new Date().toISOString(),by:state.by,rules:d.checked||[]};
    var href="/review#d="+encodeURIComponent(JSON.stringify(payload));
    var link=h("a",{class:"btn green",href:href,target:"_blank",rel:"noopener"},["Open human review checklist",icon("external")]);
    box.appendChild(h("div",{class:"human"},[
      h("div",{class:"row"},[icon("person"),h("div",{},[h("h4",{text:"Get it verified by a person",style:"font:800 18px/1.2 var(--display);margin-bottom:4px"}),h("p",{text:"An AI can miss things. Open the checklist in a new tab so a person can confirm each point and sign off."})])]),
      link
    ]));
    var t2=took(d); if(t2) box.appendChild(t2);
    show(box);
    if(window.__confetti) window.__confetti();
  }

  go.addEventListener("click",function(){
    var fd=new FormData();
    fd.append("ad",adFile);
    if(policyFile) fd.append("policy",policyFile); else fd.append("policy_text",$("policyText").value);
    go.disabled=true; go.textContent="Checking...";
    sheet.className="sheet"; stage.classList.add("scanning");
    var t0=Date.now(), msg=h("p",{text:"Reading the ad against the rules..."});
    show(h("div",{class:"working"},[h("span",{class:"spin"}),msg]));
    var timer=setInterval(function(){
      var sec=Math.round((Date.now()-t0)/1000);
      msg.textContent="Reading the ad against the rules... "+sec+"s"+(sec>25?" (taking longer than usual)":"");
    },1000);
    var ac=new AbortController(), killer=setTimeout(function(){ac.abort();},150000);

    fetch("/api/check",{method:"POST",body:fd,signal:ac.signal}).then(function(r){
      return r.json().then(function(j){return {ok:r.ok,body:j};});
    }).then(function(res){
      if(!res.ok){ show(h("p",{class:"err",text:res.body.error||"Something went wrong. Try again."})); return; }
      if(res.body.verdict==="PASS") renderPass(res.body); else renderFail(res.body);
    }).catch(function(e){
      show(h("p",{class:"err",text:(e&&e.name==="AbortError")
        ? "This is taking too long. Check your internet connection and try again."
        : "Could not reach the app. Check that it is still running, then try again."}));
    }).then(function(){
      clearInterval(timer); clearTimeout(killer);
      stage.classList.remove("scanning"); go.textContent="Check ad"; refresh();
      if(window.matchMedia("(max-width:860px)").matches){ sheet.scrollIntoView({block:"start"}); }
    });
  });

  emptyState();

  /* ---------- about: cipher wheel ---------- */
  var NS="http://www.w3.org/2000/svg", shift=3;
  function ring(g,r,size,fill){
    for(var i=0;i<26;i++){
      var t=document.createElementNS(NS,"text");
      t.setAttribute("x",160); t.setAttribute("y",160-r); t.setAttribute("text-anchor","middle");
      t.setAttribute("dominant-baseline","central"); t.setAttribute("font-size",size); t.setAttribute("fill",fill);
      t.setAttribute("transform","rotate("+(i*360/26)+" 160 160)"); t.textContent=String.fromCharCode(65+i); g.appendChild(t);
    }
  }
  ring($("outer"),132,16,"#fff"); ring($("innerLetters"),84,15,"#fff");
  function encode(s,k){
    return s.replace(/[A-Z]/g,function(c){return String.fromCharCode((c.charCodeAt(0)-65+k)%26+65);});
  }
  function spin(){
    $("inner").style.transform="rotate("+(shift*360/26)+"deg)";
    $("enc").textContent=encode("THE CIPHERERS",shift);
  }
  spin();
  $("wheelbtn").addEventListener("click",function(){ shift=(shift+1)%26; if(shift===0) shift=26; spin(); });
  $("copyMail").addEventListener("click",function(){ copy("thecipherers@gmail.com",$("copyMail"),"Copied"); });
})();
</script>
<script>
(function(){
  "use strict";
  var reduce=window.matchMedia("(prefers-reduced-motion:reduce)").matches;

  /* icons for the new graphics */
  var IC={
    shield:'<path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    image:'<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-8 9"/>',
    eye:'<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    edit:'<path d="M4 20h4L19 9a2.8 2.8 0 00-4-4L4 16v4z"/><path d="M13.5 6.5l4 4"/>',
    person:'<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0113 0"/><path d="M16 11.5l2 2 3.5-4"/>'
  };
  Array.prototype.forEach.call(document.querySelectorAll("[data-icon]"),function(el){
    var p=IC[el.getAttribute("data-icon")]; if(p) el.innerHTML='<svg viewBox="0 0 24 24">'+p+'</svg>';
  });

  /* giant wordmark, one letter at a time */
  var wm=document.getElementById("wordmark");
  if(wm){
    var label=wm.textContent; wm.setAttribute("aria-label",label); wm.textContent="";
    var idx=0;
    ["brand-","governance"].forEach(function(part){
      var w=document.createElement("span"); w.className="wd"; w.setAttribute("aria-hidden","true");
      part.split("").forEach(function(c){
        var s=document.createElement("span"); s.className="ch"+(c==="-"?" hy":"");
        s.style.setProperty("--i",idx++); s.textContent=c; w.appendChild(s);
      });
      wm.appendChild(w);
    });
  }

  /* floating shapes */
  var T={
    card:'<svg viewBox="0 0 64 64"><rect x="8" y="6" width="48" height="52" rx="8" fill="#fff" stroke="#C9C3F7" stroke-width="2"/><rect x="14" y="12" width="36" height="18" rx="4" fill="#111A36"/><rect x="14" y="36" width="26" height="4" rx="2" fill="#C9CEE0"/><rect x="14" y="44" width="18" height="4" rx="2" fill="#DCE0EE"/></svg>',
    ok:'<svg viewBox="0 0 64 64"><circle cx="32" cy="32" r="26" fill="#12805F"/><path d="M20 33l8 8 16-17" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    no:'<svg viewBox="0 0 64 64"><circle cx="32" cy="32" r="26" fill="#C7372B"/><path d="M32 18v18" stroke="#fff" stroke-width="6" stroke-linecap="round"/><circle cx="32" cy="46" r="3.6" fill="#fff"/></svg>',
    lens:'<svg viewBox="0 0 64 64"><circle cx="27" cy="27" r="17" fill="rgba(255,255,255,.55)" stroke="#4A3FE0" stroke-width="5"/><path d="M40 40l16 16" stroke="#4A3FE0" stroke-width="7" stroke-linecap="round"/></svg>',
    shield:'<svg viewBox="0 0 64 64"><path d="M32 5l21 8v15c0 15-9 25-21 31C20 53 11 43 11 28V13z" fill="#4A3FE0"/><path d="M22 31l7 7 13-15" fill="none" stroke="#FFC247" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    star:'<svg viewBox="0 0 64 64"><path d="M32 4l7 21 21 7-21 7-7 21-7-21-21-7 21-7z" fill="#FFC247"/></svg>'
  };
  var kinds=["card","ok","lens","shield","no","star","card","ok","star","lens","shield","card","no","star"];
  var box=document.getElementById("floaters");
  var count=window.innerWidth<700?7:kinds.length;
  function rnd(a,b){return a+Math.random()*(b-a);}
  var small=window.innerWidth<700;
  for(var i=0;i<count;i++){
    var d=document.createElement("div"); d.className="fl";
    var px=small?rnd(26,44):rnd(40,86);
    var edge=!small && i%5!==4;
    var left=edge?(i%2?rnd(91,98):rnd(0,5)):rnd(0,94);
    d.style.width=px+"px"; d.style.left=left+"%"; d.style.top=rnd(0,92)+"%";
    d.style.opacity=edge?rnd(.65,.9):rnd(.28,.4);
    d.style.setProperty("--d",rnd(13,28)+"s");
    d.style.setProperty("--dx",rnd(-50,50)+"px"); d.style.setProperty("--dy",rnd(-60,60)+"px");
    d.style.setProperty("--r0",rnd(-18,18)+"deg"); d.style.setProperty("--r1",rnd(-30,30)+"deg");
    d.style.animationDelay="-"+rnd(0,12)+"s";
    d.innerHTML=T[kinds[i]]; box.appendChild(d);
  }

  /* particle network that reacts to the cursor */
  var cv=document.getElementById("net"), ctx=cv.getContext("2d");
  var W=0,H=0,dpr=Math.min(window.devicePixelRatio||1,2),pts=[],mouse={x:-9999,y:-9999};
  function size(){
    W=window.innerWidth; H=window.innerHeight;
    cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
    var n=Math.round(Math.min(80,W*H/16000)); pts=[];
    for(var k=0;k<n;k++) pts.push({x:Math.random()*W,y:Math.random()*H,vx:rnd(-.35,.35),vy:rnd(-.35,.35),r:rnd(1.4,3)});
  }
  function frame(){
    ctx.clearRect(0,0,W,H);
    var j,k,p,q,dx,dy,dist;
    for(j=0;j<pts.length;j++){
      p=pts[j];
      if(!reduce){
        dx=mouse.x-p.x; dy=mouse.y-p.y; dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<170&&dist>1){ p.vx+=dx/dist*.012; p.vy+=dy/dist*.012; }
        p.vx*=.995; p.vy*=.995;
        var sp=Math.sqrt(p.vx*p.vx+p.vy*p.vy); if(sp>.9){p.vx*=.9/sp;p.vy*=.9/sp;}
        if(sp<.12){p.vx+=rnd(-.02,.02);p.vy+=rnd(-.02,.02);}
        p.x+=p.vx; p.y+=p.vy;
        if(p.x<-10)p.x=W+10; if(p.x>W+10)p.x=-10; if(p.y<-10)p.y=H+10; if(p.y>H+10)p.y=-10;
      }
    }
    for(j=0;j<pts.length;j++){
      p=pts[j];
      for(k=j+1;k<pts.length;k++){
        q=pts[k]; dx=p.x-q.x; dy=p.y-q.y; dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<130){ ctx.strokeStyle="rgba(74,63,224,"+(0.24*(1-dist/130))+")"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke(); }
      }
      dx=p.x-mouse.x; dy=p.y-mouse.y; dist=Math.sqrt(dx*dx+dy*dy);
      if(dist<190){ ctx.strokeStyle="rgba(255,160,0,"+(0.6*(1-dist/190))+")"; ctx.lineWidth=1.4; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(mouse.x,mouse.y); ctx.stroke(); }
      ctx.fillStyle="rgba(74,63,224,.55)"; ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,6.2832); ctx.fill();
    }
    if(!reduce) requestAnimationFrame(frame);
  }
  size(); frame();
  var rt; window.addEventListener("resize",function(){ clearTimeout(rt); rt=setTimeout(function(){ size(); if(reduce) frame(); },150); });
  window.addEventListener("pointermove",function(e){ mouse.x=e.clientX; mouse.y=e.clientY; });
  window.addEventListener("pointerleave",function(){ mouse.x=mouse.y=-9999; });

  /* confetti when an ad passes */
  var fx=document.getElementById("fx");
  window.__confetti=function(){
    if(reduce) return;
    var c=fx.getContext("2d"); fx.width=window.innerWidth; fx.height=window.innerHeight;
    var sh=document.getElementById("sheet").getBoundingClientRect();
    var ox=Math.max(40,Math.min(sh.left+sh.width/2,window.innerWidth-40));
    var oy=Math.max(90,Math.min(sh.top+130,window.innerHeight*.6));
    var colors=["#4A3FE0","#FFC247","#12805F","#FF6B8A","#5EE0B5","#7C6CFF"], bits=[];
    for(var n=0;n<120;n++){
      var a=-Math.PI/2+rnd(-1.15,1.15), v=rnd(6,14);
      bits.push({x:ox,y:oy,vx:Math.cos(a)*v,vy:Math.sin(a)*v,w:rnd(6,12),h:rnd(4,8),r:rnd(0,6.28),vr:rnd(-.3,.3),c:colors[n%colors.length]});
    }
    var start=null;
    function tick(t){
      if(start===null) start=t;
      var el=t-start; c.clearRect(0,0,fx.width,fx.height);
      if(el>2600){ return; }
      bits.forEach(function(b){
        b.vy+=.3; b.vx*=.992; b.x+=b.vx; b.y+=b.vy; b.r+=b.vr;
        c.save(); c.globalAlpha=Math.max(0,1-el/2600); c.translate(b.x,b.y); c.rotate(b.r); c.fillStyle=b.c; c.fillRect(-b.w/2,-b.h/2,b.w,b.h); c.restore();
      });
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  };
})();
</script>
</body>
</html>
"""

# ------------------------------------------------------ human review page
REVIEW_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human review checklist</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='9' fill='%2312805F'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Syne:wght@600;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#F4F5FB;--surface:#fff;--ink:#111A36;--muted:#5A6482;--line:#DCE0EE;--brand:#4A3FE0;--brand-soft:#E9E7FD;
  --pass:#12805F;--pass-soft:#DFF4EC;--fail:#C7372B;--focus:#1F6FEB;
  --display:"Syne",system-ui,sans-serif;--body:"Instrument Sans",system-ui,sans-serif;
  --shadow:0 1px 2px rgba(17,26,54,.06),0 14px 30px -18px rgba(74,63,224,.35)}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:400 17px/1.55 var(--body);-webkit-font-smoothing:antialiased}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.wrap{max-width:760px;margin:0 auto;padding:22px 20px 64px}
.bar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:22px}
.bar a{color:var(--ink);font-weight:600}
.brand{font:800 18px var(--display)}
h1{font:800 clamp(28px,5vw,40px)/1.08 var(--display);margin:0}
.lede{color:var(--muted);margin:10px 0 0;max-width:52ch}
.card{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow);margin-top:18px}
.meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media (max-width:620px){.meta{grid-template-columns:1fr}}
.meta small{display:block;color:var(--muted)}
.meta strong{word-break:break-word}
.chip{display:inline-block;background:var(--pass-soft);color:var(--pass);font-weight:700;border-radius:999px;padding:2px 12px}
.progress{margin-top:18px}
.progress .t{display:flex;justify-content:space-between;font-weight:600;margin-bottom:8px}
.track{height:10px;border-radius:999px;background:var(--line);overflow:hidden}
.fill{height:100%;width:0;background:var(--pass);border-radius:999px;transition:width .25s}
@media (prefers-reduced-motion:reduce){.fill{transition:none}}
h2{font:800 20px/1.2 var(--display);margin:0 0 4px}
.sub{color:var(--muted);font-size:15px;margin:0 0 12px}
.items{list-style:none;margin:0;padding:0}
.items li{border-top:1px solid var(--line)}
.items li:first-child{border-top:0}
.items label{display:flex;gap:14px;align-items:flex-start;padding:13px 0;cursor:pointer}
.items input{width:22px;height:22px;margin:2px 0 0;accent-color:var(--pass);flex:none}
.items input:checked+span{color:var(--muted);text-decoration:line-through;text-decoration-color:#9AA3BE}
.field{display:grid;gap:6px;margin-top:6px}
.field input{padding:12px;border:1px solid var(--line);border-radius:10px;font:400 17px var(--body);color:var(--ink);background:#fff;max-width:420px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:9px;background:var(--pass);color:#fff;border:2px solid var(--pass);
  border-radius:12px;padding:13px 22px;font:700 17px var(--body);cursor:pointer;text-decoration:none}
.btn:hover{background:#0E6A50;border-color:#0E6A50}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn.ghost{background:#fff;color:var(--ink);border-color:var(--line)}
.btn.ghost:hover{border-color:var(--ink)}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.hint{color:var(--muted);font-size:15px;margin-top:10px}
.seal{display:flex;gap:16px;align-items:center;border:3px double var(--pass);border-radius:14px;padding:18px 20px;margin-top:18px;background:var(--pass-soft);color:#0E6A50}
.seal svg{width:44px;height:44px;flex:none;fill:none;stroke:currentColor;stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}
.seal strong{font:800 20px/1.2 var(--display);display:block}
.empty{text-align:center;padding:60px 10px}
.empty a{color:var(--brand);font-weight:700}
@media print{
  body{background:#fff}.noprint{display:none!important}
  .card{box-shadow:none;break-inside:avoid}.wrap{padding:0}
}
</style>
</head>
<body>
<div class="wrap">
  <div class="bar noprint"><span class="brand">THE CIPHERERS</span><a href="/">Back to the checker</a></div>
  <div id="app"></div>
</div>
<script>
(function(){
  "use strict";
  var app=document.getElementById("app");
  function h(tag,props,kids){
    var n=document.createElement(tag);
    if(props){Object.keys(props).forEach(function(k){
      var v=props[k];
      if(k==="class") n.className=v; else if(k==="text") n.textContent=v;
      else if(k.slice(0,2)==="on") n.addEventListener(k.slice(2),v); else n.setAttribute(k,v);
    });}
    (kids||[]).forEach(function(c){ if(c==null) return; n.appendChild(typeof c==="string"?document.createTextNode(c):c); });
    return n;
  }
  var raw=location.hash.indexOf("#d=")===0?location.hash.slice(3):"", data=null;
  try{ data=JSON.parse(decodeURIComponent(raw)); }catch(e){}
  if(!data||!data.ad){
    app.appendChild(h("div",{class:"empty"},[
      h("h1",{text:"No checklist to show"}),
      h("p",{class:"lede",style:"margin:12px auto",text:"This page opens from a passing result. Run a check, and if the ad passes, open the human review checklist from there."}),
      h("a",{href:"/",text:"Go to the checker"})
    ]));
    document.title="Human review checklist";
    return;
  }

  var GENERAL=[
    "The wording matches the approved copy.",
    "Every claim in the ad can be backed up with proof.",
    "The logo, colors and fonts match the brand guide.",
    "Prices, dates and offer terms are correct.",
    "The fine print is present and easy to read.",
    "Contact details and links are correct."
  ];
  var items=[];
  (data.rules||[]).forEach(function(r){ items.push({g:"rules",t:"Confirm the ad follows: "+r}); });
  GENERAL.forEach(function(t){ items.push({g:"general",t:t}); });

  function hash(s){ var x=5381; for(var i=0;i<s.length;i++) x=((x<<5)+x+s.charCodeAt(i))|0; return String(x); }
  var key="review:"+hash(raw);
  var st={ticks:{},name:"",verified:false,when:""};
  try{ var saved=JSON.parse(localStorage.getItem(key)||"null"); if(saved) st=saved; }catch(e){}
  function save(){ try{localStorage.setItem(key,JSON.stringify(st));}catch(e){} }

  var when=new Date(data.at); var whenText=isNaN(when)?"":when.toLocaleString();

  app.appendChild(h("h1",{text:"Human review checklist"}));
  app.appendChild(h("p",{class:"lede",text:"The AI says this ad follows the rules. A person confirms each point below before the ad is used."}));

  app.appendChild(h("div",{class:"card"},[
    h("div",{class:"meta"},[
      h("div",{},[h("small",{text:"Ad"}),h("strong",{text:data.ad})]),
      h("div",{},[h("small",{text:"Checked on"}),h("strong",{text:whenText||"Unknown"})]),
      h("div",{},[h("small",{text:"AI result ("+(data.by||"AI")+")"}),h("span",{class:"chip",text:"PASS"})])
    ]),
    h("div",{class:"progress"},[
      h("div",{class:"t"},[h("span",{id:"ptext"}),h("span",{id:"ppct"})]),
      h("div",{class:"track"},[h("div",{class:"fill",id:"pfill"})])
    ])
  ]));

  var lists={};
  function section(title,sub,g){
    var ul=h("ul",{class:"items"}); lists[g]=ul;
    return h("section",{class:"card"},[h("h2",{text:title}),h("p",{class:"sub",text:sub}),ul]);
  }
  if((data.rules||[]).length) app.appendChild(section("Rules the AI checked","Look at the ad yourself and confirm each rule.","rules"));
  app.appendChild(section("General checks","Things an AI cannot fully judge on its own.","general"));

  items.forEach(function(it,i){
    var cb=h("input",{type:"checkbox",id:"c"+i}); cb.checked=!!st.ticks[i];
    cb.addEventListener("change",function(){ st.ticks[i]=cb.checked; st.verified=false; save(); update(); });
    lists[it.g].appendChild(h("li",{},[h("label",{},[cb,h("span",{text:it.t})])]));
  });

  var nameIn=h("input",{type:"text",id:"rname",autocomplete:"name",value:st.name||""});
  nameIn.addEventListener("input",function(){ st.name=nameIn.value; st.verified=false; save(); update(); });
  var verifyBtn=h("button",{class:"btn",type:"button",text:"Mark as verified"});
  var printBtn=h("button",{class:"btn ghost",type:"button",text:"Print or save as PDF"});
  var mailLink=h("a",{class:"btn ghost",id:"mail",text:"Email this checklist"});
  var seal=h("div",{class:"seal",id:"seal",hidden:"hidden"});
  var sign=h("section",{class:"card"},[
    h("h2",{text:"Sign-off"}),
    h("p",{class:"sub",text:"Tick every point above, then add your name."}),
    h("label",{class:"field",for:"rname"},[h("span",{text:"Reviewer name"}),nameIn]),
    h("div",{class:"actions noprint"},[verifyBtn,printBtn,mailLink]),
    seal,
    h("p",{class:"hint noprint",text:"This page doesn't send anything on its own. Print it, save it as a PDF or email it to keep a record."})
  ]);
  app.appendChild(sign);

  function done(){ return items.filter(function(_,i){return st.ticks[i];}).length; }
  function update(){
    var n=done(), total=items.length, pct=Math.round(n/total*100);
    document.getElementById("ptext").textContent=n+" of "+total+" confirmed";
    document.getElementById("ppct").textContent=pct+"%";
    document.getElementById("pfill").style.width=pct+"%";
    var ready=n===total && st.name.trim().length>0;
    verifyBtn.disabled=!ready;
    verifyBtn.textContent=st.verified?"Verified":"Mark as verified";
    seal.hidden=!st.verified;
    if(st.verified){
      seal.textContent="";
      seal.insertAdjacentHTML("afterbegin",'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/></svg>');
      seal.appendChild(h("div",{},[h("strong",{text:"Verified by "+st.name.trim()}),h("span",{text:"on "+(st.when||"")+" for "+data.ad})]));
    }
    var lines=["Human review checklist","Ad: "+data.ad,"AI result: PASS","Reviewer: "+(st.name||"(not signed)"),"Status: "+(st.verified?"Verified "+st.when:"Not yet verified"),""];
    items.forEach(function(it,i){ lines.push((st.ticks[i]?"[x] ":"[ ] ")+it.t); });
    mailLink.setAttribute("href","mailto:?subject="+encodeURIComponent("Human review: "+data.ad)+"&body="+encodeURIComponent(lines.join("\n")));
  }
  verifyBtn.addEventListener("click",function(){ st.verified=true; st.when=new Date().toLocaleString(); save(); update(); });
  printBtn.addEventListener("click",function(){ window.print(); });
  document.title="Human review: "+data.ad;
  update();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)

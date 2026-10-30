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
# Image generation (Gemini only). Change these if Google renames the models.
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
IMAGE_FALLBACK = os.environ.get("GEMINI_IMAGE_FALLBACK_MODEL", "gemini-2.5-flash-image")
IMAGE_TIMEOUT_MS = int(float(os.environ.get("IMAGE_TIMEOUT_SECONDS", "120")) * 1000)

# Languages the verdict (and the corrected ad) can be written in.
LANGUAGES = {
    "en": "English",
    "hi": "Hindi (Devanagari script)",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
    "de": "German",
    "ru": "Russian",
}

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
    "severity": "high" for legal, safety or prohibited-claim problems,
              "medium" for clear brand or policy breaks, "low" for small
              style issues
- Always list in "checked" the short names (under 12 words each, at most 12
  items) of the most important policy rules you checked the ad against.
- If something is unclear or the policy doesn't cover it, do not fail the ad
  for it.

Respond with ONLY valid JSON in this exact shape:
{"verdict": "PASS" or "FAIL",
 "checked": ["<short rule name>", "..."],
 "violations": [{"rule": "...", "reason": "...", "fix": "...", "severity": "high" or "medium" or "low"}]}
For PASS, "violations" must be an empty list."""

FIX_PROMPT = """You are an advertisement copywriter and compliance editor.

You are given (1) a company's rules and policies, (2) an advertisement that
FAILED the policy check, and (3) the list of violations found.
Your job: produce a corrected version of the ad that follows EVERY rule while
keeping the original message, offer, tone and purpose as close as possible.

Rules:
- Use ONLY the policy provided. Do not invent rules or facts.
- Keep headline, body, call_to_action and fine_print in the SAME language as the original ad.
- Do not invent prices, dates, statistics or claims that are not in the
  original ad. If a required element needs information you do not have
  (for example a legal disclaimer's exact text), use a clear placeholder in
  square brackets such as [add legal disclaimer here].
- Remove or reword every violating claim; add every missing required element.
- "design_changes" lists non-text changes a designer must make (logo, colors,
  layout, image changes). Use an empty list if there are none.
- "changes" lists, in one short line each, what you changed and which
  violation it resolves.

Respond with ONLY valid JSON in this exact shape:
{"headline": "...", "body": "...", "call_to_action": "...", "fine_print": "...",
 "design_changes": ["..."], "changes": ["..."]}
Use an empty string for any text part the ad does not need."""


def language_note(lang):
    name = LANGUAGES.get(lang, "English")
    if name == "English":
        return ""
    return (
        f"\n\nLANGUAGE: Write all human-readable text in {name}. "
        "Keep the JSON keys and the verdict value (PASS or FAIL) exactly as specified, in English."
    )


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
            "severity": str(v.get("severity", "medium")).strip().lower()
            if str(v.get("severity", "")).strip().lower() in ("high", "medium", "low")
            else "medium",
        }
        for v in violations
        if isinstance(v, dict)
    ]
    checked = [str(c).strip() for c in (result.get("checked") or []) if str(c).strip()][:12]
    if verdict == "PASS":
        clean = []
    return {"verdict": verdict, "checked": checked, "violations": clean}


def _gemini_client(timeout_ms=None):
    from google import genai
    from google.genai import types

    tm = timeout_ms or GEMINI_TIMEOUT_MS

    key = os.environ["GEMINI_API_KEY"]
    try:
        # attempts=1 turns off the library's default of 5 slow automatic retries.
        opts = types.HttpOptions(
            timeout=tm,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    except Exception:
        try:
            opts = types.HttpOptions(timeout=tm)
        except Exception:
            opts = None
    return genai.Client(api_key=key, http_options=opts) if opts else genai.Client(api_key=key)


def _gemini_call(client, model, contents, thinking, system):
    from google.genai import types

    cfg = {"system_instruction": system, "response_mime_type": "application/json"}
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


def run_gemini(policy, ad, system=SYSTEM_PROMPT, closing=None):
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
        closing or "Does the advertisement follow the policy? Reply with the JSON only.",
    ]
    client = _gemini_client()
    models = [GEMINI_MODEL] + ([GEMINI_FALLBACK] if GEMINI_FALLBACK and GEMINI_FALLBACK != GEMINI_MODEL else [])
    first_error = None
    for model in models:
        # Least thinking first: it is the biggest speed-up. Fall back if the model rejects it.
        for thinking in ("minimal", "low", None):
            try:
                return _gemini_call(client, model, contents, thinking, system)
            except Exception as e:
                first_error = first_error or e
                if thinking is not None and _is_config_error(e):
                    continue  # try the next, gentler setting on the same model
                break  # slow, busy or broken: try the faster fallback model
    raise first_error


def run_claude(policy, ad, system=SYSTEM_PROMPT, closing=None):
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
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "COMPANY POLICY DOCUMENT:"},
                block(policy),
                {"type": "text", "text": "ADVERTISEMENT TO REVIEW:"},
                block(ad),
                {"type": "text", "text": closing or "Does the advertisement follow the policy? Reply with the JSON only."},
            ],
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text")


NO_KEY = (
    "No API key is set. Set GEMINI_API_KEY (free at aistudio.google.com) "
    "or ANTHROPIC_API_KEY, then restart the app."
)


def run_check(policy, ad, lang="en"):
    which = provider()
    system = SYSTEM_PROMPT + language_note(lang)
    if which == "gemini":
        return parse_result(run_gemini(policy, ad, system=system))
    if which == "claude":
        return parse_result(run_claude(policy, ad, system=system))
    raise UserError(NO_KEY)


def fix_language_note(lang):
    name = LANGUAGES.get(lang, "English")
    if name == "English":
        return ""
    return (
        "\n\nLANGUAGE: Keep the ad copy in the original ad's language. "
        f"Write only the 'design_changes' and 'changes' lists in {name}."
    )


def parse_fix(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        r = json.loads(text)
    except json.JSONDecodeError:
        raise UserError("The AI returned an answer in an unexpected format. Try again.")
    if not isinstance(r, dict):
        raise UserError("The AI returned an answer in an unexpected format. Try again.")

    def s(key):
        return str(r.get(key) or "").strip()

    def lst(key):
        return [str(x).strip() for x in (r.get(key) or []) if str(x).strip()]

    return {
        "headline": s("headline"),
        "body": s("body"),
        "call_to_action": s("call_to_action"),
        "fine_print": s("fine_print"),
        "design_changes": lst("design_changes"),
        "changes": lst("changes"),
    }


def run_fix(policy, ad, violations, lang="en"):
    which = provider()
    system = FIX_PROMPT + fix_language_note(lang)
    closing = (
        "VIOLATIONS FOUND IN THE AD:\n"
        + json.dumps(violations, ensure_ascii=False, indent=2)
        + "\n\nWrite the corrected version of the ad. Reply with the JSON only."
    )
    if which == "gemini":
        return parse_fix(run_gemini(policy, ad, system=system, closing=closing))
    if which == "claude":
        return parse_fix(run_claude(policy, ad, system=system, closing=closing))
    raise UserError(NO_KEY)


def build_image_prompt(fix, violations, has_reference):
    lines = []
    if has_reference:
        lines.append(
            "Redesign the attached advertisement so that it follows the company's rules. Keep its overall "
            "layout, brand colors, imagery, fonts and style as close to the original as possible, "
            "changing only what is needed."
        )
    else:
        lines.append("Create a finished, professional advertisement image.")
    lines.append("Show EXACTLY this text in the ad, spelled exactly as written, and no other text:")
    for label, key in (
        ("Headline", "headline"),
        ("Body", "body"),
        ("Call-to-action button", "call_to_action"),
        ("Fine print (small text at the bottom)", "fine_print"),
    ):
        if fix.get(key):
            lines.append(f'- {label}: "{fix[key]}"')
    if fix.get("design_changes"):
        lines.append("Also make these design changes:")
        lines += [f"- {x}" for x in fix["design_changes"]]
    bad = [v for v in violations if isinstance(v, dict)]
    if bad:
        lines.append("The old version broke these rules, so none of this may appear again:")
        lines += [f"- {v.get('rule', '')}: {v.get('reason', '')}" for v in bad]
    lines.append(
        "Do not invent logos, brand names, prices, claims or badges. If a required logo is not in the "
        "reference image, leave a plain empty space for it."
    )
    lines.append("Output the finished advertisement as a single image.")
    return "\n".join(lines)


def run_image(ad, fix, violations):
    if provider() != "gemini":
        raise UserError("Image generation needs a Gemini key (GEMINI_API_KEY). Claude models cannot generate images.")
    from google.genai import types

    has_ref = ad[0] == "file" and str(ad[2]).startswith("image/")
    contents = [build_image_prompt(fix, violations, has_ref)]
    if has_ref:
        contents.append(types.Part.from_bytes(data=ad[1], mime_type=ad[2]))
    client = _gemini_client(IMAGE_TIMEOUT_MS)
    models = [IMAGE_MODEL] + ([IMAGE_FALLBACK] if IMAGE_FALLBACK and IMAGE_FALLBACK != IMAGE_MODEL else [])
    first_error = None
    text_reply = ""
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
        except Exception as e:
            first_error = first_error or e
            continue
        for part in getattr(response, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                mime = inline.mime_type or "image/png"
                data = base64.b64encode(inline.data).decode()
                return {"image": f"data:{mime};base64,{data}", "mime": mime, "model": model}
            if getattr(part, "text", None):
                text_reply = part.text
    if first_error and not text_reply:
        raise first_error
    raise UserError("The AI did not return an image." + (f" It said: {text_reply[:200]}" if text_reply else " Try again."))


def read_policy_from_request():
    policy_file = request.files.get("policy")
    policy_text = (request.form.get("policy_text") or "").strip()
    if policy_file and policy_file.filename:
        return read_upload(policy_file, policy=True)
    if policy_text:
        return ("text", policy_text)
    raise UserError("Add the company rules first, as a file or as pasted text.")


def lang_from_request():
    lang = (request.form.get("lang") or "en").strip()
    return lang if lang in LANGUAGES else "en"


# ------------------------------------------------------------------ routes
@app.get("/api/status")
def status():
    which = provider()
    name = "brand-governance"
    by = {"gemini": "Google Gemini", "claude": "Claude"}.get(which, "AI")
    return jsonify({"provider": which, "name": name, "by": by, "languages": LANGUAGES, "image": which == "gemini"})


@app.post("/api/check")
def check():
    try:
        ad_file = request.files.get("ad")
        if not ad_file or not ad_file.filename:
            raise UserError("Add the advertisement first.")

        policy = read_policy_from_request()
        ad = read_upload(ad_file)
        started = time.time()
        result = run_check(policy, ad, lang_from_request())
        result["seconds"] = round(time.time() - started, 1)
        print(f"[check] {provider()} answered in {result['seconds']}s", flush=True)
        return jsonify(result)
    except UserError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # provider/network errors
        return jsonify({"error": f"The check could not be completed: {str(e)[:300]}"}), 502


@app.post("/api/fix")
def fix():
    """Generate a corrected version of a failed ad."""
    try:
        ad_file = request.files.get("ad")
        if not ad_file or not ad_file.filename:
            raise UserError("Add the advertisement first.")
        try:
            violations = json.loads(request.form.get("violations") or "[]")
        except json.JSONDecodeError:
            violations = []
        if not isinstance(violations, list) or not violations:
            raise UserError("There are no violations to fix. Run a check first.")
        policy = read_policy_from_request()
        ad = read_upload(ad_file)
        started = time.time()
        result = run_fix(policy, ad, violations, lang_from_request())
        result["seconds"] = round(time.time() - started, 1)
        print(f"[fix] {provider()} answered in {result['seconds']}s", flush=True)
        return jsonify(result)
    except UserError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"The corrected ad could not be generated: {str(e)[:300]}"}), 502


@app.post("/api/fix-image")
def fix_image():
    """Generate an actual corrected ad image (Gemini image model)."""
    try:
        ad_file = request.files.get("ad")
        if not ad_file or not ad_file.filename:
            raise UserError("Add the advertisement first.")
        try:
            fix = json.loads(request.form.get("fix") or "{}")
            violations = json.loads(request.form.get("violations") or "[]")
        except json.JSONDecodeError:
            raise UserError("Generate the corrected ad text first.")
        if not isinstance(fix, dict) or not any(fix.get(k) for k in ("headline", "body", "call_to_action")):
            raise UserError("Generate the corrected ad text first.")
        if not isinstance(violations, list):
            violations = []
        ad = read_upload(ad_file)
        started = time.time()
        result = run_image(ad, fix, violations)
        result["seconds"] = round(time.time() - started, 1)
        print(f"[image] {result['model']} answered in {result['seconds']}s", flush=True)
        return jsonify(result)
    except UserError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            return jsonify({"error": (
                "Image generation is out of quota on this Gemini key. Free keys have little or no "
                "image quota, and it resets daily. Try again later, or enable billing on the key. "
                "The corrected ad text above still works."
            )}), 429
        return jsonify({"error": (
            f"The ad image could not be generated: {msg[:250]} "
            "(If the model name is wrong or unavailable on your key, set GEMINI_IMAGE_MODEL.)"
        )}), 502


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
LOGO_PNG = base64.b64decode("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")


@app.get("/logo.png")
def logo():
    return Response(LOGO_PNG, mimetype="image/png", headers={"Cache-Control": "public, max-age=86400"})


PAGE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>brand-governance</title>
<link rel="icon" type="image/png" href="/logo.png">
<script>
try{var m=localStorage.getItem("bg-theme")||"dark";var d=m==="dark"||(m==="system"&&matchMedia("(prefers-color-scheme:dark)").matches);document.documentElement.setAttribute("data-theme",d?"dark":"light");}catch(e){}
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080921;--glow1:rgba(109,93,246,.30);--glow2:rgba(56,92,255,.20);
  --side:rgba(13,14,42,.78);--panel:rgba(21,23,62,.78);--panel2:rgba(30,32,82,.62);--field:rgba(8,9,32,.55);
  --line:rgba(140,140,255,.17);--ink:#F2F3FF;--muted:#9EA3CF;--brand:#6D5DF6;--brand2:#8E80FF;
  --pass:#2ED8A3;--pass-bg:rgba(46,216,163,.15);--fail:#FF5F73;--fail-bg:rgba(255,95,115,.15);--warn:#FFB454;--warn-bg:rgba(255,180,84,.15);
  --shadow:0 24px 50px -26px rgba(0,0,0,.75);--focus:#9C90FF;
  --display:"Plus Jakarta Sans","Inter",system-ui,"Noto Sans Devanagari","Noto Sans SC","Noto Sans JP","Yu Gothic","Microsoft YaHei",sans-serif;
  --body:"Inter",system-ui,"Noto Sans Devanagari","Noto Sans SC","Noto Sans JP","Yu Gothic","Microsoft YaHei",sans-serif;
  color-scheme:dark}
[data-theme=light]{
  --bg:#F3F3FD;--glow1:rgba(109,93,246,.15);--glow2:rgba(110,150,255,.13);
  --side:rgba(255,255,255,.85);--panel:#FFFFFF;--panel2:#F5F5FF;--field:#F7F7FF;
  --line:#E3E3F4;--ink:#15163A;--muted:#6A6D98;--brand:#6D5DF6;--brand2:#5A49E8;
  --pass:#0E9F74;--pass-bg:#DDF6EC;--fail:#D6384C;--fail-bg:#FDE5E8;--warn:#B96A00;--warn-bg:#FFF0D6;
  --shadow:0 18px 40px -28px rgba(80,70,200,.5);--focus:#5A49E8;color-scheme:light}
*{box-sizing:border-box}
[hidden]{display:none!important}
html{scroll-behavior:smooth}
body{margin:0;color:var(--ink);font:400 14.5px/1.5 var(--body);-webkit-font-smoothing:antialiased;min-height:100vh;
  background:radial-gradient(900px 520px at 88% -8%,var(--glow1),transparent 62%),radial-gradient(760px 520px at -6% 34%,var(--glow2),transparent 62%),var(--bg);background-attachment:fixed}
h1,h2,h3,h4{font-family:var(--display);font-weight:800;margin:0;letter-spacing:-.02em;color:var(--ink);text-wrap:balance}
p{margin:0}
a{color:var(--brand2)}
button{font:inherit;color:inherit;cursor:pointer}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.vh{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.ic{display:inline-flex;width:1.15em;height:1.15em;flex:none;vertical-align:-.2em}
.ic svg{width:100%;height:100%;fill:none;stroke:currentColor;stroke-width:1.9;stroke-linecap:round;stroke-linejoin:round}

/* ---------- shell ---------- */
.app{display:grid;grid-template-columns:236px minmax(0,1fr);grid-template-areas:"top top" "side main";min-height:100vh}
.top{grid-area:top;display:flex;align-items:center;gap:14px;padding:12px 22px 12px 18px;position:sticky;top:0;z-index:30;
  background:var(--side);-webkit-backdrop-filter:blur(16px);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}
.brandb{display:flex;align-items:center;gap:10px;width:206px;flex:none;text-decoration:none;color:var(--ink);font:800 17px var(--display);letter-spacing:-.02em}

.search{position:relative;flex:1;max-width:560px}
.search input{width:100%;height:40px;border-radius:12px;border:1px solid var(--line);background:var(--field);color:var(--ink);padding:0 64px 0 40px;font:inherit}
.search input::placeholder{color:var(--muted)}
.search .si{position:absolute;left:13px;top:11px;color:var(--muted)}
.kbd{position:absolute;right:10px;top:9px;font:600 11px var(--body);color:var(--muted);border:1px solid var(--line);border-radius:7px;padding:3px 7px;background:var(--panel2)}
.spacer{flex:1}
.themesw{display:flex;border:1px solid var(--line);border-radius:999px;padding:3px;background:var(--field);gap:2px}
.themesw button{border:0;background:transparent;width:34px;height:30px;border-radius:999px;color:var(--muted);display:grid;place-items:center}
.themesw button[aria-pressed=true]{background:var(--brand);color:#fff;box-shadow:0 6px 16px -6px var(--brand)}
.lang{display:flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;background:var(--field);padding:0 6px 0 12px;height:40px;color:var(--muted)}
.lang select{border:0;background:transparent;color:var(--ink);font:600 13px var(--body);height:36px;cursor:pointer;max-width:120px}
.lang select option{color:#15163A;background:#fff}
.iconbtn{position:relative;width:40px;height:40px;border-radius:12px;border:1px solid var(--line);background:var(--field);display:grid;place-items:center;color:var(--ink)}
.iconbtn .dot{position:absolute;top:8px;right:9px;width:9px;height:9px;border-radius:50%;background:var(--fail);border:2px solid var(--bg)}
.upgrade{display:flex;align-items:center;gap:8px;height:40px;padding:0 16px;border-radius:12px;border:0;background:linear-gradient(135deg,#6D5DF6,#8B5CF6);color:#fff;font-weight:700;box-shadow:0 12px 26px -12px #6D5DF6}
.upgrade:hover{filter:brightness(1.08)}

.side{grid-area:side;position:sticky;top:65px;align-self:start;height:calc(100vh - 65px);display:flex;flex-direction:column;padding:16px 12px 14px;
  background:var(--side);-webkit-backdrop-filter:blur(16px);backdrop-filter:blur(16px);border-right:1px solid var(--line)}
.nav{display:grid;gap:4px}
.nav a{display:flex;align-items:center;gap:12px;padding:11px 14px;border-radius:12px;color:var(--muted);text-decoration:none;font-weight:600}
.nav a:hover{background:var(--panel2);color:var(--ink)}
.nav a.on{background:linear-gradient(135deg,rgba(109,93,246,.95),rgba(139,92,246,.85));color:#fff;box-shadow:0 12px 26px -14px #6D5DF6}
.nav .ic{width:19px;height:19px}
.sidefoot{margin-top:auto;display:grid;gap:10px}
.user{display:flex;align-items:center;gap:11px;padding:10px 12px;border-top:1px solid var(--line);margin-top:4px}
.avatar{width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#6D5DF6,#FF8FB1);display:grid;place-items:center;color:#fff;font:800 14px var(--display);flex:none}
.user b{display:block;font-size:13px;line-height:1.2}
.user small{color:var(--muted);font-size:12px}

.main{grid-area:main;padding:26px 28px 40px;min-width:0}
.page{max-width:1380px;margin:0 auto}
.phead{margin-bottom:18px}
.phead h1{font-size:clamp(24px,3vw,32px);line-height:1.1}
.sub{color:var(--muted);margin-top:6px;max-width:60ch}
.sub a{font-weight:600;text-decoration:none}
.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--shadow);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px)}
.card>h3,.cardh{font-size:16px;margin-bottom:12px}
.cardh{display:flex;justify-content:space-between;align-items:center;gap:10px}
.cardh a,.linkbtn{font:600 13px var(--body);color:var(--brand2);text-decoration:none;background:none;border:0;padding:0}
.cardh a:hover,.linkbtn:hover{text-decoration:underline}

/* ---------- buttons ---------- */
.btn,.btnp{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:42px;padding:0 18px;border-radius:12px;font-weight:700;text-decoration:none;border:1px solid var(--line);background:var(--panel2);color:var(--ink)}
.btn:hover{border-color:var(--brand2)}
.btnp{background:linear-gradient(135deg,#6D5DF6,#8B5CF6);color:#fff;border-color:transparent;box-shadow:0 14px 28px -14px #6D5DF6}
.btnp:hover{filter:brightness(1.08)}
.btn:disabled,.btnp:disabled{opacity:.45;cursor:not-allowed;box-shadow:none;filter:none}
.btn.green{background:var(--pass);border-color:transparent;color:#04281d}
.btn.gen{width:100%;margin-top:16px;background:linear-gradient(135deg,#6D5DF6,#8B5CF6);color:#fff;border-color:transparent}
.btn.block,.btnp.block{width:100%;min-height:48px;font-size:15px}
.btn.small{min-height:34px;padding:0 12px;font-size:13px;border-radius:10px}
.link{background:none;border:0;color:var(--brand2);font-weight:600;padding:6px 0;text-decoration:underline}

/* ---------- badges ---------- */
.badge{display:inline-flex;align-items:center;gap:6px;padding:4px 11px;border-radius:999px;font:700 12px var(--body);white-space:nowrap}
.badge.pass{background:var(--pass-bg);color:var(--pass)}
.badge.fail{background:var(--fail-bg);color:var(--fail)}
.badge.high{background:var(--fail-bg);color:var(--fail)}
.badge.medium{background:var(--warn-bg);color:var(--warn)}
.badge.low{background:var(--panel2);color:var(--muted);border:1px solid var(--line)}
.badge.soon{background:var(--warn-bg);color:var(--warn)}
.tag{display:inline-block;padding:3px 9px;border-radius:8px;background:var(--panel2);border:1px solid var(--line);color:var(--muted);font-size:12px;font-weight:600;margin-right:6px}

/* ---------- home ---------- */
.homegrid{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:20px;align-items:start}
.homeL{display:grid;gap:20px;min-width:0}
.uprow{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:20px}
.upload{position:relative;overflow:hidden}
.dz{display:grid;grid-template-columns:minmax(0,1fr) 120px;gap:14px;border:2px dashed rgba(140,130,255,.45);border-radius:16px;padding:22px 20px;margin-top:16px;background:linear-gradient(180deg,rgba(109,93,246,.06),transparent);transition:.15s}
.dz.over{border-color:var(--brand2);background:rgba(109,93,246,.14)}
.dz .mid{display:grid;justify-items:center;text-align:center;gap:6px;align-content:center}
.dz .cube{width:54px;height:54px;border-radius:16px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(109,93,246,.35),rgba(139,92,246,.15));color:var(--brand2);margin-bottom:4px;font-size:26px}
.dz strong{font:800 15px var(--display)}
.dz small{color:var(--muted);font-size:12px}
.chips{display:grid;gap:8px;align-content:center}
.chip{text-align:center;padding:7px 0;border-radius:10px;border:1px solid var(--line);background:var(--panel2);color:var(--muted);font:700 12px var(--body);letter-spacing:.04em}
.or{color:var(--muted);font-size:12px;margin:2px 0}
.split{position:relative;display:inline-flex}
.split .btnp{border-top-right-radius:0;border-bottom-right-radius:0}
.split .caret{width:40px;min-height:42px;border:0;border-left:1px solid rgba(255,255,255,.3);border-radius:0 12px 12px 0;background:linear-gradient(135deg,#7B6BFA,#9568F8);color:#fff;display:grid;place-items:center}
.menu{position:absolute;z-index:40;top:calc(100% + 6px);left:0;min-width:210px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px;box-shadow:var(--shadow);-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px)}
.menu button{display:flex;width:100%;align-items:center;gap:10px;border:0;background:none;padding:9px 11px;border-radius:9px;text-align:left;font-weight:600}
.menu button:hover{background:var(--panel2)}
.urlrow{display:flex;gap:0;margin-top:10px;width:100%}
.urlrow input{flex:1;min-width:0;height:42px;border-radius:12px 0 0 12px;border:1px solid var(--line);border-right:0;background:var(--field);color:var(--ink);padding:0 14px;font:inherit}
.urlrow button{width:46px;border:1px solid var(--line);border-radius:0 12px 12px 0;background:var(--panel2);color:var(--ink);display:grid;place-items:center}
.deco i{position:absolute;display:block;border-radius:14px;background:linear-gradient(135deg,#8B7BFF,#6D5DF6);box-shadow:0 18px 30px -12px rgba(109,93,246,.8);animation:bob 6s ease-in-out infinite}
@keyframes bob{50%{transform:translateY(-8px) rotate(4deg)}}
.rlist{display:grid;gap:10px}
.rrow{display:flex;align-items:center;gap:12px;padding:8px;border-radius:12px}
.rrow:hover{background:var(--panel2)}
.rrow .nm{font-weight:700;font-size:13.5px;line-height:1.25}
.rrow small{display:block;color:var(--muted);font-size:12px}
.rrow .grow{flex:1;min-width:0}
.rrow button{all:unset;cursor:pointer;display:flex;gap:12px;align-items:center;flex:1;min-width:0}
.thumb{width:46px;height:46px;border-radius:11px;flex:none;object-fit:cover;display:grid;place-items:center;color:rgba(255,255,255,.9);overflow:hidden}
.thumb svg{width:22px;height:22px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
.stat{padding:16px 16px 14px;position:relative;overflow:hidden}
.stat small{color:var(--muted);font-weight:600;font-size:12.5px}
.stat b{display:block;font:800 30px/1.15 var(--display);margin-top:4px;letter-spacing:-.03em}
.stat .d{display:block;font:700 12px var(--body);margin-top:2px;letter-spacing:0}
.stat .d.up{color:var(--pass)}.stat .d.down{color:var(--fail)}
.stat .spark{position:absolute;right:14px;bottom:14px;display:flex;align-items:flex-end;gap:3px;height:38px}
.spark i{display:block;width:5px;border-radius:2px;opacity:.9}
.tbl{display:grid}
.trow{display:grid;grid-template-columns:minmax(0,2fr) 90px 150px minmax(0,1.4fr) 140px;gap:14px;align-items:center;padding:12px 6px;border-top:1px solid var(--line)}
.trow.head{border-top:0;color:var(--muted);font:600 12px var(--body);padding-top:2px}
.trow .who{display:flex;gap:12px;align-items:center;min-width:0}
.trow .who b{display:block;font-size:14px}
.trow .who small{color:var(--muted);font-size:12px}
.issuesfound{color:var(--fail);font-weight:600;display:flex;align-items:center;gap:6px}
.issuesfound.ok{color:var(--pass)}
.vr{position:relative;display:inline-block}
.vr>button{display:flex;align-items:center;gap:8px;min-height:36px;padding:0 12px;border-radius:10px;border:1px solid var(--line);background:var(--panel2);font-weight:600;font-size:13px}
.vr>button:hover{border-color:var(--brand2)}

/* verdict panel */
.vpanel{position:sticky;top:90px}
.vtop{display:flex;justify-content:space-between;align-items:center;gap:12px}
.vtop .badge{font-size:13px;padding:7px 14px}
.vsum{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:12px 0 14px}
.vsum p{color:var(--muted);font-size:12.5px;line-height:1.45;flex:1}
.ring{position:relative;width:84px;height:84px;flex:none}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring .t{position:absolute;inset:0;display:grid;place-items:center;text-align:center;font:800 22px/1 var(--display)}
.ring .t small{display:block;font:600 11px var(--body);color:var(--muted);margin-top:2px}
.vimg{height:150px;border-radius:14px;overflow:hidden;background:var(--panel2);border:1px solid var(--line);display:grid;place-items:center}
.vimg img{width:100%;height:100%;object-fit:contain;background:#0b0c24}
.vimg svg{width:100%;height:100%}
.tabs2{display:flex;gap:4px;margin:14px 0 8px;border-bottom:1px solid var(--line)}
.tabs2 button{border:0;background:none;padding:9px 12px;color:var(--muted);font-weight:600;font-size:13px;border-bottom:2px solid transparent;margin-bottom:-1px}
.tabs2 button[aria-selected=true]{color:var(--ink);border-color:var(--brand2)}
.vbox{max-height:300px;overflow:auto;display:grid;gap:8px;padding-right:2px}
.iss{display:flex;gap:11px;padding:10px;border-radius:12px;background:var(--panel2);border:1px solid var(--line)}
.iss .mk{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;flex:none;font-size:15px;background:var(--fail-bg);color:var(--fail)}
.iss .mk.ok{background:var(--pass-bg);color:var(--pass)}
.iss .mk.warn{background:var(--warn-bg);color:var(--warn)}
.iss .tx{flex:1;min-width:0}
.iss b{display:block;font-size:13px;line-height:1.3}
.iss span.r{display:block;color:var(--muted);font-size:12px;margin-top:2px;line-height:1.4}
.iss .badge{align-self:flex-start;font-size:11px;padding:3px 9px}
.vfoot{margin-top:14px;display:grid}
.emptyv{text-align:center;color:var(--muted);padding:26px 10px;display:grid;gap:12px;justify-items:center}

/* ---------- check page ---------- */
.checkgrid{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:20px;align-items:start}
.stack{display:grid;gap:16px}
.drop{display:grid;gap:5px;justify-items:center;text-align:center;border:2px dashed rgba(140,130,255,.45);border-radius:16px;padding:26px 16px;cursor:pointer;transition:.15s;background:linear-gradient(180deg,rgba(109,93,246,.06),transparent)}
.drop:hover,.drop.over{border-color:var(--brand2);background:rgba(109,93,246,.14)}
.drop.has-file{border-style:solid;border-color:var(--pass)}
.drop .ic{font-size:26px;color:var(--brand2)}
.drop strong{font:700 14px var(--display);word-break:break-word}
.drop small{color:var(--muted)}
.stage{position:relative;margin-top:14px;border-radius:14px;overflow:hidden;border:1px solid var(--line);background:var(--field)}
.stage img{display:block;width:100%;max-height:340px;object-fit:contain}
.stage.scanning::after{content:"";position:absolute;left:0;right:0;height:56px;top:0;background:linear-gradient(180deg,transparent,rgba(142,128,255,.5),transparent);animation:scan 1.5s linear infinite}
@keyframes scan{from{transform:translateY(-60px)}to{transform:translateY(380px)}}
.pdfcard{display:flex;align-items:center;gap:12px;padding:16px;font-size:15px}
.pdfcard .ic{font-size:30px;color:var(--brand2)}
.pdfcard span{display:block;color:var(--muted);font-size:12px}
.rulesline{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.rulesline .ri{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:var(--panel2);color:var(--brand2);font-size:20px;flex:none}
.rulesline .rt{flex:1;min-width:180px}
.rulesline .rt b{display:block}
.rulesline .rt small{color:var(--muted)}
.rulesline.warn .ri{background:var(--warn-bg);color:var(--warn)}
.banner{margin:0 0 16px;padding:12px 16px;border-radius:12px;background:var(--warn-bg);color:var(--warn);font-weight:600;border:1px solid var(--line)}
.sheet.fail{border-color:var(--fail)}
.sheet.pass{border-color:var(--pass)}
.empty{display:grid;justify-items:center;gap:12px;text-align:center;color:var(--muted);padding:34px 12px}
.empty svg{width:96px;opacity:.9}
.working{display:flex;align-items:center;gap:14px;padding:18px 4px;color:var(--muted)}
.spin{width:22px;height:22px;border-radius:50%;border:3px solid var(--line);border-top-color:var(--brand2);animation:sp 1s linear infinite;flex:none}
@keyframes sp{to{transform:rotate(360deg)}}
.err{color:var(--fail);font-weight:600;padding:12px 4px}
.stampwrap{margin:2px 0 10px}
.stamp{display:inline-block;font:800 36px/1 var(--display);letter-spacing:.06em;border:4px solid;border-radius:12px;padding:9px 20px;transform:rotate(-3deg)}
.stamp.long{font-size:clamp(22px,3.6vw,32px);letter-spacing:.03em;padding:8px 16px}
.stamp.pass{color:var(--pass);border-color:var(--pass)}
.stamp.fail{color:var(--fail);border-color:var(--fail)}
.slam{animation:slam .45s cubic-bezier(.2,1.5,.4,1) both}
@keyframes slam{from{transform:rotate(-3deg) scale(2.2);opacity:0}to{transform:rotate(-3deg) scale(1);opacity:1}}
.vtitle{font-size:21px;line-height:1.2}
.vsub{color:var(--muted);margin:6px 0 14px}
.vlist,.checked{list-style:none;margin:0;padding:0;display:grid;gap:10px}
.vcard{padding:14px;border-radius:14px;background:var(--panel2);border:1px solid var(--line);border-left:4px solid var(--fail)}
.vcard h4{font-size:14.5px;margin-bottom:4px}
.vcard .wrong{color:var(--muted);font-size:13.5px}
.fixbox{display:flex;gap:10px;margin-top:10px;padding:10px 12px;border-radius:10px;background:var(--pass-bg);color:var(--ink);font-size:13.5px}
.fixbox .ic{color:var(--pass);margin-top:2px}
.checked li{display:flex;gap:10px;align-items:center;color:var(--ink)}
.checked .ic{color:var(--pass)}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.fine{color:var(--muted);font-size:12.5px;margin-top:12px}
.human{margin-top:18px;padding:16px;border-radius:14px;background:var(--pass-bg);display:grid;gap:12px}
.human .row{display:flex;gap:12px}
.human .row .ic{font-size:24px;color:var(--pass);margin-top:2px}
.human p{color:var(--muted);font-size:13.5px}
.fixed{margin-top:18px;border:1px solid var(--brand2);border-radius:16px;padding:16px;background:var(--panel2)}
.fixed-h{font-size:17px;margin-bottom:12px}
.adcard{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;display:grid;gap:10px}
.ad-h{font-size:19px;line-height:1.2}
.ad-b{white-space:pre-wrap}
.ad-cta{justify-self:start;background:var(--brand);color:#fff;font-weight:700;border-radius:999px;padding:7px 16px}
.ad-fine{color:var(--muted);font-size:12.5px;white-space:pre-wrap}
.fl-h{font-size:14px;margin:14px 0 6px}
.fixlist{margin:0;padding-left:20px;display:grid;gap:5px}
.imgbox{margin-top:14px}
.imgbox img{display:block;max-width:100%;border-radius:12px;border:1px solid var(--line);margin-top:10px}

/* ---------- rules / history / reports / integrations / settings ---------- */
.rulesgrid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:20px;align-items:start}
.sep{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:12.5px;margin:16px 0 10px}
.sep::before,.sep::after{content:"";flex:1;height:1px;background:var(--line)}
textarea{width:100%;min-height:200px;resize:vertical;border-radius:12px;border:1px solid var(--line);background:var(--field);color:var(--ink);padding:14px;font:inherit;line-height:1.55}
textarea::placeholder{color:var(--muted)}
.tips{display:grid;gap:10px;margin:12px 0 0;padding:0;list-style:none;color:var(--muted)}
.tips li{display:flex;gap:10px}
.tips .ic{color:var(--brand2);margin-top:2px}
.savedtag{color:var(--pass);font-weight:600;font-size:13px;min-height:20px;margin-top:8px}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
.toolbar input{height:40px;border-radius:12px;border:1px solid var(--line);background:var(--field);color:var(--ink);padding:0 14px;font:inherit;min-width:220px;flex:1;max-width:340px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:12px;padding:3px;background:var(--field);gap:2px}
.seg button{border:0;background:none;padding:7px 14px;border-radius:9px;font-weight:600;color:var(--muted)}
.seg button[aria-pressed=true]{background:var(--brand);color:#fff}
.rep{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.bars{display:grid;gap:12px}
.bar{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,3fr) 34px;gap:10px;align-items:center;font-size:13px}
.bar .track{height:10px;border-radius:99px;background:var(--panel2);overflow:hidden}
.bar .fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#6D5DF6,#8B7BFF)}
.bar .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar b{text-align:right}
.week{display:flex;align-items:flex-end;gap:10px;height:160px;padding-top:8px}
.week div{flex:1;display:grid;justify-items:center;gap:6px;height:100%;align-content:end}
.week i{display:block;width:100%;max-width:40px;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#8B7BFF,#6D5DF6)}
.week span{font-size:11.5px;color:var(--muted)}
.week em{font-style:normal;font:700 11px var(--body);color:var(--ink)}
.donutwrap{display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.legend{display:grid;gap:8px}
.legend span{display:flex;align-items:center;gap:8px}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block}
.integ{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px}
.integ .card{display:grid;gap:12px;align-content:start}
.integ .lg{width:46px;height:46px;border-radius:13px;display:grid;place-items:center;color:#fff;font:800 18px var(--display)}
.integ p{color:var(--muted);font-size:13.5px}
.integ .top2{display:flex;justify-content:space-between;align-items:flex-start}
.setrow{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:16px 0;border-top:1px solid var(--line);flex-wrap:wrap}
.setrow:first-of-type{border-top:0}
.setrow b{display:block}
.setrow small{color:var(--muted)}
.switch{position:relative;width:46px;height:26px;border-radius:99px;border:0;background:var(--panel2);border:1px solid var(--line);flex:none}
.switch::after{content:"";position:absolute;top:2px;left:2px;width:20px;height:20px;border-radius:50%;background:var(--muted);transition:.15s}
.switch[aria-checked=true]{background:var(--brand);border-color:var(--brand)}
.switch[aria-checked=true]::after{left:22px;background:#fff}
.steps{counter-reset:s;list-style:none;margin:0 0 18px;padding:0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.steps li{counter-increment:s;background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px}
.steps li::before{content:counter(s);display:grid;place-items:center;width:34px;height:34px;border-radius:50%;background:var(--brand);color:#fff;font:800 15px var(--display);margin-bottom:12px}
.steps h3{font-size:16px;margin-bottom:6px}
.steps p{color:var(--muted)}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-bottom:18px}
.two .card h3{margin-bottom:6px}
.two .card p,.know li{color:var(--muted)}
.know ul{margin:8px 0 0;padding-left:20px;display:grid;gap:6px}
.about{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:30px;align-items:center}
.wheelbtn{all:unset;cursor:pointer;display:block;width:100%;max-width:320px;border-radius:50%}
.wheelbtn:focus-visible{outline:3px solid var(--focus);outline-offset:4px}
.wheel{width:100%;height:auto;display:block;filter:drop-shadow(0 20px 30px rgba(0,0,0,.35))}
.wheel #inner{transition:transform .5s cubic-bezier(.3,1.4,.5,1)}
.cipher-line{margin-top:14px;color:var(--muted)}
.about h2{font-size:34px;margin-bottom:10px}
.about .desc{color:var(--muted);max-width:56ch}
.contact{display:flex;align-items:center;gap:14px;margin-top:20px;padding:14px;border-radius:14px;background:var(--panel2);border:1px solid var(--line);flex-wrap:wrap}
.contact .ic{font-size:24px;color:var(--brand2)}
.contact .who{flex:1;min-width:180px}
.contact small{display:block;color:var(--muted)}

/* ---------- overlays ---------- */
.overlay{position:fixed;inset:0;z-index:80;background:rgba(4,4,20,.62);display:grid;place-items:center;padding:20px;-webkit-backdrop-filter:blur(4px);backdrop-filter:blur(4px)}
.modal{width:min(560px,100%);max-height:88vh;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 40px 80px -30px rgba(0,0,0,.8)}
.modal.wide{width:min(760px,100%)}
.modal .mh{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:14px}
.modal .x{width:36px;height:36px;border-radius:10px;border:1px solid var(--line);background:var(--panel2);display:grid;place-items:center;flex:none}
.soon{display:grid;justify-items:center;text-align:center;gap:12px;padding:10px 6px 4px}
.soon .big{width:64px;height:64px;border-radius:20px;display:grid;place-items:center;background:linear-gradient(135deg,#6D5DF6,#8B5CF6);color:#fff;font-size:30px;box-shadow:0 18px 34px -14px #6D5DF6}
.soon h2{font-size:30px}
.soon p{color:var(--muted);max-width:42ch}
.rep-sec{margin-top:18px}
.rep-sec h4{font-size:14px;margin-bottom:8px}
.pop{position:absolute;z-index:60;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:8px;box-shadow:var(--shadow);-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px)}
.pop.bell{right:0;top:calc(100% + 10px);width:330px}
.pop.sres{left:0;right:0;top:calc(100% + 8px);max-height:340px;overflow:auto}
.pop .ph{font:700 12px var(--body);color:var(--muted);padding:8px 10px 4px;text-transform:uppercase;letter-spacing:.05em}
.pop .it{display:flex;gap:11px;align-items:center;width:100%;border:0;background:none;text-align:left;padding:9px 10px;border-radius:10px}
.pop .it:hover{background:var(--panel2)}
.pop .it small{display:block;color:var(--muted);font-size:12px}
.pop .none{padding:16px;color:var(--muted);text-align:center}
.relw{position:relative}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:90;background:var(--ink);color:var(--bg);padding:11px 18px;border-radius:12px;font-weight:600;box-shadow:0 20px 40px -16px rgba(0,0,0,.6);max-width:90vw}
canvas#fx{position:fixed;inset:0;pointer-events:none;z-index:70}

/* ---------- responsive ---------- */
@media (max-width:1240px){.homegrid{grid-template-columns:minmax(0,1fr)}.vpanel{position:static}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:980px){
  .app{grid-template-columns:minmax(0,1fr);grid-template-areas:"top" "side" "main"}
  .side{position:static;height:auto;flex-direction:row;padding:8px 12px;border-right:0;border-bottom:1px solid var(--line);overflow-x:auto}
  .nav{display:flex;gap:4px}
  .nav a{white-space:nowrap;padding:9px 12px}
  .sidefoot{display:none}
  .uprow,.checkgrid,.rulesgrid,.rep,.about,.two,.steps{grid-template-columns:minmax(0,1fr)}
  .main{padding:18px 14px 36px}
  .brandb{width:auto}.brandb span{display:none}
  .trow{grid-template-columns:minmax(0,1fr) 90px 130px}
  .trow>:nth-child(3),.trow>:nth-child(4){display:none}
  .trow.head>:nth-child(3),.trow.head>:nth-child(4){display:none}
}
@media (max-width:640px){.search{display:none}.spacer{display:block}.top{gap:8px;padding:10px 12px}.upgrade span{display:none}.upgrade{padding:0 12px}.dz{grid-template-columns:minmax(0,1fr)}.chips{grid-template-columns:repeat(4,1fr)}.deco{display:none}.stats{grid-template-columns:minmax(0,1fr)}.lang{padding-left:8px}.lang .gl{display:none}.themesw button{width:30px}}
/* ---------- logo + motion ---------- */
.app{position:relative;z-index:1}
canvas#net{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none}
.bgfx{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.bgfx i{position:absolute;border-radius:50%;filter:blur(70px);opacity:.55}
.bgfx .o1{width:46vw;height:46vw;left:-12vw;top:-10vw;background:radial-gradient(circle,rgba(109,93,246,.55),transparent 65%);animation:drift1 22s ease-in-out infinite}
.bgfx .o2{width:40vw;height:40vw;right:-10vw;top:10vh;background:radial-gradient(circle,rgba(46,180,255,.38),transparent 65%);animation:drift2 26s ease-in-out infinite}
.bgfx .o3{width:36vw;height:36vw;left:30vw;bottom:-18vw;background:radial-gradient(circle,rgba(255,120,190,.30),transparent 65%);animation:drift3 30s ease-in-out infinite}
[data-theme=light] .bgfx i{opacity:.4}
@keyframes drift1{50%{transform:translate(8vw,6vw) scale(1.12)}}
@keyframes drift2{50%{transform:translate(-7vw,8vw) scale(1.1)}}
@keyframes drift3{50%{transform:translate(6vw,-6vw) scale(1.15)}}
.logo{position:relative;width:46px;height:46px;flex:none;border-radius:50%;display:grid;place-items:center}
.logo::before{content:"";position:absolute;inset:-3px;border-radius:50%;background:conic-gradient(from 0deg,#6D5DF6,#8E80FF,#2ED8A3,#FFC247,#FF6B8A,#6D5DF6);animation:spinr 7s linear infinite}
.logo::after{content:"";position:absolute;inset:-3px;border-radius:50%;box-shadow:0 0 20px rgba(109,93,246,.75);animation:pulsel 3.2s ease-in-out infinite}
.logo img{position:relative;z-index:1;width:100%;height:100%;border-radius:50%;background:#fff;padding:1px;animation:stampin .8s cubic-bezier(.2,1.4,.4,1) both}
.brandb:hover .logo img{animation:stampin .6s cubic-bezier(.2,1.4,.4,1) both}
@keyframes spinr{to{transform:rotate(360deg)}}
@keyframes pulsel{50%{box-shadow:0 0 34px rgba(142,128,255,.95)}}
@keyframes stampin{0%{transform:scale(1.9) rotate(-18deg);opacity:0}60%{transform:scale(.94) rotate(3deg);opacity:1}100%{transform:scale(1) rotate(0)}}
.card{animation:fadeUp .6s cubic-bezier(.2,.7,.2,1) backwards;transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}}
.stat:hover,.integ .card:hover,.steps li:hover{transform:translateY(-4px)}
.stat:hover,.integ .card:hover{border-color:var(--brand2);box-shadow:0 22px 40px -22px rgba(109,93,246,.7)}
.steps li{transition:transform .25s ease}
.spark i{transform-origin:bottom;animation:growy .8s cubic-bezier(.2,.8,.2,1) backwards;animation-delay:calc(var(--i,0)*45ms)}
@keyframes growy{from{transform:scaleY(0)}}
.ring .arc{transition:stroke-dasharray 1.2s cubic-bezier(.2,.8,.2,1)}
.btnp:not(:disabled),.upgrade{position:relative;overflow:hidden}
.btnp:not(:disabled)::after,.upgrade::after{content:"";position:absolute;top:0;left:-70%;width:40%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.4),transparent);transform:skewX(-20deg);animation:shine 4s ease-in-out infinite;pointer-events:none}
@keyframes shine{0%,55%{left:-70%}100%{left:140%}}
.btnp:hover,.upgrade:hover{transform:translateY(-1px)}
.dz,.drop{transition:border-color .2s,background .2s,box-shadow .3s}
.dz{animation:dzglow 3.4s ease-in-out infinite}
@keyframes dzglow{50%{box-shadow:0 0 0 6px rgba(109,93,246,.10),0 0 34px -6px rgba(109,93,246,.45)}}
.dz .cube{animation:bob 4s ease-in-out infinite}
.chip{transition:transform .2s,border-color .2s,color .2s}
.dz:hover .chip{transform:translateX(-4px);border-color:var(--brand2);color:var(--ink)}
.deco{position:absolute;right:30px;bottom:80px;width:90px;height:76px;opacity:.95;pointer-events:none}
.iconbtn .dot{animation:ping 2s ease-out infinite}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(255,95,115,.7)}80%,100%{box-shadow:0 0 0 9px rgba(255,95,115,0)}}
.nav a{transition:background .2s,color .2s,transform .2s}
.nav a:hover{transform:translateX(3px)}
.nav a.on{animation:navglow 3s ease-in-out infinite}
@keyframes navglow{50%{box-shadow:0 14px 34px -10px #8B5CF6}}
.badge.fail{animation:failpulse 2.6s ease-in-out infinite}
@keyframes failpulse{50%{box-shadow:0 0 0 4px rgba(255,95,115,.12)}}
.vimg{position:relative}
.vimg::after{content:"";position:absolute;left:0;right:0;top:0;height:46px;background:linear-gradient(180deg,transparent,rgba(142,128,255,.35),transparent);animation:scan2 3.6s ease-in-out infinite;pointer-events:none}
@keyframes scan2{0%{transform:translateY(-50px)}100%{transform:translateY(190px)}}
.themesw button{transition:background .2s,color .2s,box-shadow .2s}
.themesw button[aria-pressed=true] .ic{animation:spinr .6s ease}
@media (max-width:640px){.trow{grid-template-columns:minmax(0,1fr) auto}.trow>:nth-child(5){grid-column:1/-1}.trow.head{display:none}.trow .thumb{width:38px;height:38px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<div class="bgfx" aria-hidden="true"><i class="o1"></i><i class="o2"></i><i class="o3"></i></div>
<canvas id="net" aria-hidden="true"></canvas>
<canvas id="fx" aria-hidden="true"></canvas>
<div class="app">
  <header class="top">
    <a class="brandb" href="#home" aria-label="brand-governance home">
      <span class="logo"><img src="/logo.png" alt="" width="40" height="40"></span>
      <span id="appname">brand-governance</span>
    </a>
    <div class="search relw">
      <span class="si ic" data-i="search" aria-hidden="true"></span>
      <input id="q" type="search" placeholder="Search verifications, ads, or rules..." autocomplete="off" aria-label="Search verifications, ads, or rules">
      <span class="kbd" aria-hidden="true">Ctrl K</span>
      <div class="pop sres" id="qres" hidden></div>
    </div>
    <span class="spacer"></span>
    <div class="themesw" role="group" aria-label="Color theme">
      <button type="button" id="thLight" aria-pressed="false" title="Light mode" aria-label="Light mode"><span class="ic" data-i="sun"></span></button>
      <button type="button" id="thDark" aria-pressed="true" title="Dark mode" aria-label="Dark mode"><span class="ic" data-i="moon"></span></button>
    </div>
    <label class="lang" title="Verdict language">
      <span class="ic gl" data-i="globe" aria-hidden="true"></span>
      <select id="lang" aria-label="Verdict language">
        <option value="en">English</option>
        <option value="hi">हिन्दी (Hindi)</option>
        <option value="zh">中文 (Chinese)</option>
        <option value="ja">日本語 (Japanese)</option>
        <option value="de">Deutsch (German)</option>
        <option value="ru">Русский (Russian)</option>
      </select>
    </label>
    <div class="relw">
      <button class="iconbtn" id="bellBtn" type="button" aria-label="Notifications" aria-haspopup="true"><span class="ic" data-i="bell"></span><span class="dot" id="bellDot" hidden></span></button>
      <div class="pop bell" id="bellPop" hidden></div>
    </div>
    <button class="upgrade" id="upgradeBtn" type="button"><span class="ic" data-i="sparkle"></span><span>Upgrade</span></button>
  </header>

  <aside class="side">
    <nav class="nav" aria-label="Main">
      <a href="#home" id="nav-home"><span class="ic" data-i="home"></span>Home</a>
      <a href="#check" id="nav-check"><span class="ic" data-i="shield"></span>Check Ad</a>
      <a href="#history" id="nav-history"><span class="ic" data-i="clock"></span>Past Verifications</a>
      <a href="#rules" id="nav-rules"><span class="ic" data-i="rules"></span>Company Rules</a>
      <a href="#team" id="nav-team"><span class="ic" data-i="team"></span>Team</a>
      <a href="#reports" id="nav-reports"><span class="ic" data-i="reports"></span>Reports</a>
      <a href="#integrations" id="nav-integrations"><span class="ic" data-i="plug"></span>Integrations</a>
    </nav>
    <div class="sidefoot">
      <nav class="nav" aria-label="Settings"><a href="#settings" id="nav-settings"><span class="ic" data-i="gear"></span>Settings</a></nav>
      <div class="user"><span class="avatar" aria-hidden="true">YS</span><div><b>Yashasvi Singh</b><small>Workspace Admin</small></div></div>
    </div>
  </aside>

  <main class="main" id="main">

  <!-- HOME -->
  <section class="page" id="p-home" hidden>
    <div class="homegrid">
      <div class="homeL">
        <div class="uprow">
          <section class="card upload">
            <div class="deco" aria-hidden="true"><i style="width:40px;height:40px;right:0;top:0"></i><i style="width:28px;height:28px;right:54px;top:30px;animation-delay:-2s"></i><i style="width:16px;height:16px;right:14px;top:56px;animation-delay:-4s;border-radius:50%"></i></div>
            <h1 style="font-size:clamp(24px,2.6vw,30px);line-height:1.1">Verify your advertisement</h1>
            <p class="sub" style="font-size:13.5px">Upload your ad and let our AI check it against your company's brand and advertising policies.</p>
            <p style="margin-top:8px"><a class="linkbtn" href="#how"><span class="ic" data-i="info"></span> How it works?</a></p>
            <label class="dz" id="homeDrop">
              <input class="vh" type="file" id="homeFile" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp">
              <span class="mid">
                <span class="cube"><span class="ic" data-i="upload"></span></span>
                <strong>Drag &amp; drop your ad here</strong>
                <small>Supports images and PDFs (max 20 MB)</small>
              </span>
              <span class="chips" aria-hidden="true"><span class="chip">JPG</span><span class="chip">PNG</span><span class="chip">PDF</span><span class="chip">WEBP</span></span>
            </label>
            <div style="display:grid;justify-items:start;gap:6px;margin-top:14px">
              <div class="split relw" id="splitUp">
                <button class="btnp" id="homeUp" type="button"><span class="ic" data-i="upload"></span>Upload Ad</button>
                <button class="caret" id="homeCaret" type="button" aria-label="More upload options" aria-haspopup="true"><span class="ic" data-i="chev"></span></button>
                <div class="menu" id="upMenu" hidden>
                  <button type="button" id="mDevice"><span class="ic" data-i="upload"></span>From this device</button>
                  <button type="button" id="mUrl"><span class="ic" data-i="link"></span>From a URL <span class="badge soon" style="margin-left:auto">Soon</span></button>
                </div>
              </div>
              <span class="or">or</span>
              <div class="urlrow"><input id="urlIn" type="url" placeholder="Paste a URL to an ad" aria-label="Paste a URL to an ad"><button type="button" id="urlGo" aria-label="Import from URL"><span class="ic" data-i="arrow"></span></button></div>
            </div>
          </section>
          <section class="card">
            <div class="cardh"><h3 style="margin:0">Recent Verifications</h3><a href="#history">View all <span class="ic" data-i="arrow"></span></a></div>
            <div class="rlist" id="recentList" style="margin-top:12px"></div>
          </section>
        </div>
        <div class="stats" id="stats"></div>
        <section class="card">
          <div class="cardh"><h3 style="margin:0">Latest Verifications</h3><a href="#history">View all <span class="ic" data-i="arrow"></span></a></div>
          <div class="tbl" id="latestTbl" style="margin-top:8px"></div>
        </section>
      </div>
      <aside class="card vpanel" id="vpanel" aria-live="polite"></aside>
    </div>
  </section>

  <!-- CHECK -->
  <section class="page" id="p-check" hidden>
    <div class="phead"><h1>Check an ad</h1><p class="sub">Add your ad, then check it against your saved company rules. You get a verdict and clear steps to fix anything that breaks.</p></div>
    <div class="banner" id="banner" role="alert" hidden></div>
    <div class="checkgrid">
      <div class="stack">
        <section class="card">
          <h3>Advertisement</h3>
          <label class="drop" id="adDrop">
            <input class="vh" type="file" id="adFile" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp">
            <span class="ic" data-i="upload" aria-hidden="true"></span>
            <strong id="adName">Drop the ad here, or click to choose</strong>
            <small>PDF, PNG, JPG, GIF or WEBP (max 20 MB)</small>
          </label>
          <button class="link" id="adClear" type="button" hidden>Remove file</button>
          <div class="stage" id="stage" hidden></div>
        </section>
        <section class="card rulesline" id="rulesStatus"></section>
        <button class="btnp block" id="go" type="button" disabled>Check ad</button>
      </div>
      <section class="card sheet" id="sheet" aria-labelledby="vh" aria-live="polite">
        <h3 id="vh">Verdict</h3>
        <div id="result"></div>
      </section>
    </div>
  </section>

  <!-- HISTORY -->
  <section class="page" id="p-history" hidden>
    <div class="phead"><h1>Past Verifications</h1><p class="sub">Every ad you have checked, newest first.</p></div>
    <div class="toolbar">
      <input id="hq" type="search" placeholder="Filter by ad name" aria-label="Filter by ad name">
      <div class="seg" id="hseg" role="group" aria-label="Filter by result">
        <button type="button" data-f="all" aria-pressed="true">All</button>
        <button type="button" data-f="PASS" aria-pressed="false">Pass</button>
        <button type="button" data-f="FAIL" aria-pressed="false">Fail</button>
      </div>
      <span class="spacer"></span>
      <button class="btn small" id="csvBtn" type="button"><span class="ic" data-i="download"></span>Export CSV</button>
    </div>
    <section class="card"><div class="tbl" id="histTbl"></div></section>
  </section>

  <!-- RULES -->
  <section class="page" id="p-rules" hidden>
    <div class="phead"><h1>Company Rules</h1><p class="sub">The policy every ad is checked against. It is saved in this browser, so you only add it once.</p></div>
    <div class="rulesgrid">
      <section class="card">
        <h3>Your policy</h3>
        <label class="drop" id="policyDrop">
          <input class="vh" type="file" id="policyFile" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.txt,.md">
          <span class="ic" data-i="upload" aria-hidden="true"></span>
          <strong id="policyName">Drop the rules file here, or click to choose</strong>
          <small>PDF, image, TXT or MD</small>
        </label>
        <button class="link" id="policyClear" type="button" hidden>Remove file</button>
        <div class="sep">or paste the rules</div>
        <textarea id="policyText" placeholder="For example: No claims of guaranteed results. The company logo must appear." aria-label="Paste company rules"></textarea>
        <p class="savedtag" id="savedTag" aria-live="polite"></p>
      </section>
      <section class="card">
        <h3>Tips</h3>
        <ul class="tips">
          <li><span class="ic" data-i="check"></span>Write one rule per line so each can be checked on its own.</li>
          <li><span class="ic" data-i="check"></span>Be specific: "logo at least 10% of the width" beats "logo must be visible".</li>
          <li><span class="ic" data-i="check"></span>The checker only uses the rules you give it. It adds none of its own.</li>
        </ul>
        <div class="actions"><button class="btn" id="sampleRules" type="button"><span class="ic" data-i="sparkle"></span>Use sample rules</button><a class="btnp" href="#check">Go to Check Ad</a></div>
      </section>
    </div>
  </section>

  <!-- REPORTS -->
  <section class="page" id="p-reports" hidden>
    <div class="phead"><h1>Reports</h1><p class="sub">How your ads are doing against the rules.</p></div>
    <div class="stats" id="repStats" style="margin-bottom:18px"></div>
    <div class="rep">
      <section class="card"><h3>Pass vs fail</h3><div class="donutwrap" id="repDonut"></div></section>
      <section class="card"><h3>Checks in the last 7 days</h3><div class="week" id="repWeek"></div></section>
      <section class="card"><h3>Issues by severity</h3><div class="bars" id="repSev"></div></section>
      <section class="card"><h3>Most broken rules</h3><div class="bars" id="repTop"></div></section>
    </div>
  </section>

  <!-- INTEGRATIONS -->
  <section class="page" id="p-integrations" hidden>
    <div class="phead"><h1>Integrations</h1><p class="sub">Connect brand-governance to the tools your team already uses.</p></div>
    <div class="integ" id="integGrid"></div>
  </section>

  <!-- TEAM -->
  <section class="page" id="p-team" hidden>
    <div class="phead"><h1>Team</h1><p class="sub">The people behind brand-governance.</p></div>
    <section class="card">
      <div class="about">
        <div>
          <button class="wheelbtn" id="wheelbtn" type="button" aria-label="Turn the cipher wheel by one letter">
            <svg class="wheel" viewBox="0 0 320 320" aria-hidden="true">
              <circle cx="160" cy="160" r="154" fill="#111A36"/>
              <g id="outer"></g>
              <g id="inner" style="transform-origin:160px 160px"><circle cx="160" cy="160" r="108" fill="#6D5DF6"/><g id="innerLetters"></g></g>
              <circle cx="160" cy="160" r="52" fill="#FFC247"/>
              <text x="160" y="154" text-anchor="middle" font-size="8.5" fill="#111A36" style="font-weight:800">THE</text>
              <text x="160" y="167" text-anchor="middle" font-size="8.5" fill="#111A36" style="font-weight:800">CIPHERERS</text>
              <path d="M151 4h18l-9 18z" fill="#FFC247"/>
            </svg>
          </button>
          <p class="cipher-line">Turn the wheel. THE CIPHERERS becomes <strong id="enc" aria-live="polite">WKH FLSKHUHUV</strong></p>
        </div>
        <div>
          <h2>THE CIPHERERS</h2>
          <p class="desc">The team behind brand-governance. We built it so you can find out fast whether an ad follows your company's rules, and know exactly what to change when it doesn't.</p>
          <div class="contact"><span class="ic" data-i="mail"></span><div class="who"><small>Contact us</small><a href="mailto:thecipherers@gmail.com">thecipherers@gmail.com</a></div><button class="btn small" id="copyMail" type="button">Copy email</button></div>
        </div>
      </div>
    </section>
  </section>

  <!-- SETTINGS -->
  <section class="page" id="p-settings" hidden>
    <div class="phead"><h1>Settings</h1><p class="sub">Appearance, sample data and saved information.</p></div>
    <section class="card" style="max-width:820px">
      <div class="setrow"><div><b>Appearance</b><small>Choose how brand-governance looks.</small></div>
        <div class="seg" id="themeSeg" role="group" aria-label="Appearance">
          <button type="button" data-m="light" aria-pressed="false">Light</button>
          <button type="button" data-m="dark" aria-pressed="true">Dark</button>
          <button type="button" data-m="system" aria-pressed="false">System</button>
        </div></div>
      <div class="setrow"><div><b>Show sample data</b><small>Fills the dashboard with example verifications and stats. Turn it off to see only your own checks.</small></div>
        <button class="switch" id="demoSw" type="button" role="switch" aria-checked="true" aria-label="Show sample data"></button></div>
      <div class="setrow"><div><b>AI provider</b><small id="provInfo">Checking...</small></div></div>
      <div class="setrow"><div><b>Verdict language</b><small>Use the globe menu in the top bar to change the language of verdicts.</small></div></div>
      <div class="setrow"><div><b>Clear my verification history</b><small>Removes the checks saved in this browser. Your saved rules stay.</small></div>
        <button class="btn" id="clearHist" type="button">Clear history</button></div>
      <div class="setrow"><div><b>Remove saved company rules</b><small>Deletes the rules saved in this browser.</small></div>
        <button class="btn" id="clearRules" type="button">Remove rules</button></div>
    </section>
  </section>

  <!-- HOW -->
  <section class="page" id="p-how" hidden>
    <div class="phead"><h1>How it works</h1><p class="sub">Three steps, then you know where your ad stands.</p></div>
    <ol class="steps">
      <li><h3>Add the rules</h3><p>Upload your company's policy, or paste the rules once under Company Rules.</p></li>
      <li><h3>Add the ad</h3><p>Upload a PDF or an image. The checker reads the text, logos, claims and fine print.</p></li>
      <li><h3>Read the verdict</h3><p>You get a pass, or a list of what to change, with a score and how serious each issue is.</p></li>
    </ol>
    <div class="two">
      <div class="card"><h3>If the ad fails</h3><p>Each broken rule comes with what is wrong and how to fix it. The app can also write a corrected version of the ad for you. Change the ad, upload the new version and check again until it passes.</p></div>
      <div class="card"><h3>If the ad passes</h3><p>You can open a checklist in a new tab for a person to review and sign off. Print it, save it as a PDF or email it.</p></div>
    </div>
    <section class="card know"><h3>Good to know</h3><ul>
      <li>The checker only judges the ad against the rules you give it. It adds no rules of its own.</li>
      <li>An AI can miss things. That is why a pass comes with a checklist for a person to confirm.</li>
      <li>Your files are sent to the AI service connected to this app to be read. Don't upload anything you are not allowed to share.</li>
    </ul></section>
  </section>

  </main>
</div>
<div class="toast" id="toast" role="status" hidden></div>
<div id="overlayHost"></div>
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
  function lsGet(k,d){ try{ var v=localStorage.getItem(k); return v==null?d:v; }catch(e){ return d; } }
  function lsSet(k,v){ try{ localStorage.setItem(k,v); return true; }catch(e){ return false; } }
  function lsDel(k){ try{ localStorage.removeItem(k); }catch(e){} }

  /* ---------- icons ---------- */
  var P={
    home:'<path d="M3 11l9-8 9 8"/><path d="M5 10v10h5v-6h4v6h5V10"/>',
    shield:'<path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    rules:'<path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6z"/><path d="M9 10h6M9 13.5h4"/>',
    clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    team:'<circle cx="9" cy="8" r="3.2"/><path d="M3 20a6 6 0 0112 0"/><circle cx="17" cy="9" r="2.5"/><path d="M16 14.2A5 5 0 0121 19"/>',
    reports:'<path d="M5 20V10M12 20V4M19 20v-7"/>',
    plug:'<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><path d="M16.5 13v7M13 16.5h7"/>',
    gear:'<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1.1 1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z"/>',
    bell:'<path d="M6 8a6 6 0 1112 0c0 7 3 8 3 8H3s3-1 3-8"/><path d="M10.3 20a2 2 0 003.4 0"/>',
    search:'<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
    sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon:'<path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/>',
    globe:'<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.7 4 5.7 4 9s-1.4 6.3-4 9c-2.6-2.7-4-5.7-4-9s1.4-6.3 4-9z"/>',
    chev:'<path d="M6 9l6 6 6-6"/>',
    arrow:'<path d="M5 12h14M13 6l6 6-6 6"/>',
    x:'<path d="M6 6l12 12M18 6L6 18"/>',
    link:'<path d="M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1"/>',
    info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/>',
    check:'<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    warn:'<path d="M12 8v5M12 16.5v.01"/><path d="M10.3 3.9L2.5 17.5a2 2 0 001.7 3h15.6a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/>',
    edit:'<path d="M4 20h4L19 9a2.8 2.8 0 00-4-4L4 16v4z"/><path d="M13.5 6.5l4 4"/>',
    file:'<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/>',
    image:'<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-8 9"/>',
    external:'<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5"/>',
    mail:'<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3.5 7l8.5 6.5L20.5 7"/>',
    person:'<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0113 0"/><path d="M16 11.5l2 2 3.5-4"/>',
    upload:'<path d="M12 16V5M7.5 9.5L12 5l4.5 4.5"/><path d="M5 19h14"/>',
    download:'<path d="M12 4v11M7.5 10.5L12 15l4.5-4.5"/><path d="M5 19h14"/>',
    sparkle:'<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/>'
  };
  function svgOf(name){ return '<svg viewBox="0 0 24 24" aria-hidden="true">'+(P[name]||"")+'</svg>'; }
  function icon(name){ var s=document.createElement("span"); s.className="ic"; s.setAttribute("aria-hidden","true"); s.innerHTML=svgOf(name); return s; }
  Array.prototype.forEach.call(document.querySelectorAll("[data-i]"),function(el){ el.innerHTML=svgOf(el.getAttribute("data-i")); });

  /* ---------- toast + overlays ---------- */
  var toastT=null;
  function toast(msg){
    var t=$("toast"); t.textContent=msg; t.hidden=false; clearTimeout(toastT);
    toastT=setTimeout(function(){ t.hidden=true; },3200);
  }
  var lastFocus=null;
  function openModal(node,wide){
    closeModal();
    lastFocus=document.activeElement;
    var x=h("button",{class:"x",type:"button","aria-label":"Close",onclick:closeModal},[icon("x")]);
    node.querySelector(".mh")?node.querySelector(".mh").appendChild(x):node.insertBefore(h("div",{class:"mh"},[h("span"),x]),node.firstChild);
    var m=h("div",{class:"modal"+(wide?" wide":""),role:"dialog","aria-modal":"true"},[node]);
    var o=h("div",{class:"overlay",id:"ovl"},[m]);
    o.addEventListener("mousedown",function(e){ if(e.target===o) closeModal(); });
    $("overlayHost").appendChild(o);
    x.focus();
  }
  function closeModal(){
    var o=$("ovl"); if(o){ o.parentNode.removeChild(o); if(lastFocus&&lastFocus.focus) try{lastFocus.focus();}catch(e){} }
  }
  function closePops(){ $("bellPop").hidden=true; $("qres").hidden=true; $("upMenu").hidden=true; Array.prototype.forEach.call(document.querySelectorAll(".vr .menu"),function(m){m.remove();}); }
  document.addEventListener("keydown",function(e){
    if(e.key==="Escape"){ closeModal(); closePops(); }
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="k"){ e.preventDefault(); $("q").focus(); $("q").select(); }
  });
  document.addEventListener("click",function(e){
    if(!e.target.closest(".relw")&&!e.target.closest(".vr")) closePops();
  });

  function showUpgrade(){
    var node=h("div",{},[
      h("div",{class:"mh"},[h("span")]),
      h("div",{class:"soon"},[
        h("span",{class:"big"},[icon("sparkle")]),
        h("span",{class:"badge soon",text:"Upgrade"}),
        h("h2",{text:"Coming soon"}),
        h("p",{text:"Paid plans are not available yet. We are working on them and will announce them here."}),
        h("button",{class:"btnp",type:"button",text:"Got it",onclick:closeModal})
      ])
    ]);
    openModal(node);
  }
  $("upgradeBtn").addEventListener("click",showUpgrade);

  /* ---------- theme ---------- */
  var themeMode=lsGet("bg-theme","dark");
  function isDark(){ return themeMode==="dark"||(themeMode==="system"&&window.matchMedia("(prefers-color-scheme: dark)").matches); }
  function applyTheme(){
    var d=isDark();
    document.documentElement.setAttribute("data-theme",d?"dark":"light");
    $("thDark").setAttribute("aria-pressed",d?"true":"false");
    $("thLight").setAttribute("aria-pressed",d?"false":"true");
    Array.prototype.forEach.call($("themeSeg").querySelectorAll("button"),function(b){ b.setAttribute("aria-pressed",b.getAttribute("data-m")===themeMode?"true":"false"); });
  }
  function setTheme(m){ themeMode=m; lsSet("bg-theme",m); applyTheme(); }
  $("thDark").addEventListener("click",function(){ setTheme("dark"); });
  $("thLight").addEventListener("click",function(){ setTheme("light"); });
  Array.prototype.forEach.call($("themeSeg").querySelectorAll("button"),function(b){ b.addEventListener("click",function(){ setTheme(b.getAttribute("data-m")); }); });
  try{ window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",function(){ if(themeMode==="system") applyTheme(); }); }catch(e){}
  applyTheme();

  /* ---------- verdict language ---------- */
  var I18N={
    en:{PASS:"PASS",FAIL:"FAIL",failOne:"Fix 1 thing, then check again",failMany:"Fix {n} things, then check again",failNone:"This ad breaks the company's rules",failSub:"Each card shows the rule, what is wrong in your ad, and how to fix it.",howFix:"How to fix: ",defFix:"Change the ad so it no longer breaks this rule, then check again.",copyFix:"Copy fix list",copied:"Copied",again:"Upload the fixed ad",passTitle:"This ad follows the company's rules",looked:"Rules the checker looked at:",humanH:"Get it verified by a person",humanP:"An AI can miss things. Open the checklist in a new tab so a person can confirm each point and sign off.",openList:"Open human review checklist",took:"Checked in {s} seconds.",reading:"Reading the ad against the rules...",slow:"(taking longer than usual)",checking:"Checking...",checkBtn:"Check ad",gen:"Generate corrected ad",genning:"Writing a corrected ad...",fixedH:"Corrected ad",headline:"Headline",body:"Body text",cta:"Call to action",fine:"Fine print",design:"Design changes needed",changes:"What was changed",copyAd:"Copy corrected ad",dl:"Download .txt",regen:"Generate again",aiNote:"Written by AI from your rules. A person should review it before it is used.",rerun:"The language was changed. Press Check ad again to get the verdict in this language.",problem:"Problem",fix:"Fix",adLabel:"Ad",resLabel:"Result",img:"AI-generated corrected ad",imgWorking:"Generating the corrected ad image... this can take up to a minute",imgDl:"Download image",imgRegen:"Regenerate image",imgNote:"AI-generated image. Check every word in it carefully before using it."},
    hi:{PASS:"पास",FAIL:"फेल",failOne:"1 बात ठीक करें, फिर दोबारा जाँचें",failMany:"{n} बातें ठीक करें, फिर दोबारा जाँचें",failNone:"यह विज्ञापन कंपनी के नियमों का उल्लंघन करता है",failSub:"हर कार्ड में नियम, आपके विज्ञापन में क्या गलत है और उसे कैसे ठीक करें, यह दिया गया है।",howFix:"ठीक कैसे करें: ",defFix:"विज्ञापन में ऐसा बदलाव करें कि वह इस नियम का उल्लंघन न करे, फिर दोबारा जाँचें।",copyFix:"सुधार सूची कॉपी करें",copied:"कॉपी हो गया",again:"सुधारा हुआ विज्ञापन अपलोड करें",passTitle:"यह विज्ञापन कंपनी के नियमों का पालन करता है",looked:"जाँचकर्ता ने ये नियम देखे:",humanH:"किसी व्यक्ति से सत्यापित करवाएँ",humanP:"AI से चूक हो सकती है। नए टैब में चेकलिस्ट खोलें ताकि कोई व्यक्ति हर बिंदु की पुष्टि करके हस्ताक्षर कर सके।",openList:"मानव समीक्षा चेकलिस्ट खोलें",took:"{s} सेकंड में जाँच पूरी हुई।",reading:"नियमों के अनुसार विज्ञापन पढ़ा जा रहा है...",slow:"(सामान्य से अधिक समय लग रहा है)",checking:"जाँच हो रही है...",checkBtn:"विज्ञापन जाँचें",gen:"सुधारा हुआ विज्ञापन बनाएँ",genning:"सुधारा हुआ विज्ञापन लिखा जा रहा है...",fixedH:"सुधारा हुआ विज्ञापन",headline:"शीर्षक",body:"मुख्य पाठ",cta:"कॉल टू एक्शन",fine:"छोटे अक्षरों की शर्तें",design:"ज़रूरी डिज़ाइन बदलाव",changes:"क्या बदला गया",copyAd:"सुधारा विज्ञापन कॉपी करें",dl:".txt डाउनलोड करें",regen:"फिर से बनाएँ",aiNote:"यह AI ने आपके नियमों के आधार पर लिखा है। उपयोग से पहले किसी व्यक्ति को इसकी समीक्षा करनी चाहिए।",rerun:"भाषा बदली गई है। इस भाषा में नतीजा पाने के लिए फिर से \"विज्ञापन जाँचें\" दबाएँ।",problem:"समस्या",fix:"सुधार",adLabel:"विज्ञापन",resLabel:"नतीजा",img:"AI से बना सुधारा हुआ विज्ञापन",imgWorking:"सुधारे हुए विज्ञापन की तस्वीर बन रही है... इसमें एक मिनट तक लग सकता है",imgDl:"तस्वीर डाउनलोड करें",imgRegen:"तस्वीर फिर से बनाएँ",imgNote:"यह AI से बनी तस्वीर है। उपयोग से पहले इसके हर शब्द को ध्यान से जाँचें।"},
    zh:{PASS:"通过",FAIL:"未通过",failOne:"请修改 1 处，然后重新检查",failMany:"请修改 {n} 处，然后重新检查",failNone:"该广告违反了公司规定",failSub:"每张卡片显示相关规定、广告中的问题以及修改方法。",howFix:"修改方法：",defFix:"修改广告使其不再违反此规定，然后重新检查。",copyFix:"复制修改清单",copied:"已复制",again:"上传修改后的广告",passTitle:"该广告符合公司规定",looked:"检查器查看的规定：",humanH:"请人工核实",humanP:"AI 可能会有疏漏。请在新标签页中打开清单，由人工逐项确认并签字。",openList:"打开人工审核清单",took:"检查用时 {s} 秒。",reading:"正在对照规定阅读广告...",slow:"（耗时比平时长）",checking:"检查中...",checkBtn:"检查广告",gen:"生成修正后的广告",genning:"正在撰写修正后的广告...",fixedH:"修正后的广告",headline:"标题",body:"正文",cta:"行动号召",fine:"细则说明",design:"需要的设计修改",changes:"修改内容",copyAd:"复制修正后的广告",dl:"下载 .txt",regen:"重新生成",aiNote:"由 AI 根据您的规定撰写，使用前请由人工审核。",rerun:"语言已更改。请再次点击“检查广告”以用此语言获取结果。",problem:"问题",fix:"修改",adLabel:"广告",resLabel:"结果",img:"AI 生成的修正后广告",imgWorking:"正在生成修正后的广告图片...最长可能需要一分钟",imgDl:"下载图片",imgRegen:"重新生成图片",imgNote:"此图片由 AI 生成，使用前请仔细检查其中每个字。"},
    ja:{PASS:"合格",FAIL:"不合格",failOne:"1 件を修正してから、もう一度チェックしてください",failMany:"{n} 件を修正してから、もう一度チェックしてください",failNone:"この広告は社内ルールに違反しています",failSub:"各カードに、ルール、広告のどこが問題か、修正方法が表示されます。",howFix:"修正方法：",defFix:"このルールに違反しないよう広告を修正し、もう一度チェックしてください。",copyFix:"修正リストをコピー",copied:"コピーしました",again:"修正した広告をアップロード",passTitle:"この広告は社内ルールに適合しています",looked:"チェックしたルール：",humanH:"担当者による確認を行いましょう",humanP:"AI は見落とすことがあります。新しいタブでチェックリストを開き、担当者が各項目を確認して承認してください。",openList:"人による確認チェックリストを開く",took:"{s} 秒でチェックしました。",reading:"ルールに照らして広告を確認しています...",slow:"（通常より時間がかかっています）",checking:"チェック中...",checkBtn:"広告をチェック",gen:"修正版の広告を生成",genning:"修正版の広告を作成中...",fixedH:"修正版の広告",headline:"見出し",body:"本文",cta:"行動喚起",fine:"注意書き",design:"必要なデザイン変更",changes:"変更点",copyAd:"修正版をコピー",dl:".txt をダウンロード",regen:"もう一度生成",aiNote:"AI があなたのルールに基づいて作成しました。使用前に担当者が確認してください。",rerun:"言語が変更されました。この言語で結果を得るには、もう一度「広告をチェック」を押してください。",problem:"問題",fix:"修正",adLabel:"広告",resLabel:"結果",img:"AI が生成した修正版広告",imgWorking:"修正版の広告画像を生成中...最大 1 分ほどかかります",imgDl:"画像をダウンロード",imgRegen:"画像を再生成",imgNote:"AI 生成画像です。使用前に文字をひとつずつ確認してください。"},
    de:{PASS:"BESTANDEN",FAIL:"DURCHGEFALLEN",failOne:"1 Punkt beheben, dann erneut prüfen",failMany:"{n} Punkte beheben, dann erneut prüfen",failNone:"Diese Anzeige verstößt gegen die Unternehmensregeln",failSub:"Jede Karte zeigt die Regel, was in Ihrer Anzeige falsch ist und wie Sie es beheben.",howFix:"So beheben Sie es: ",defFix:"Ändern Sie die Anzeige so, dass sie diese Regel nicht mehr verletzt, und prüfen Sie erneut.",copyFix:"Korrekturliste kopieren",copied:"Kopiert",again:"Korrigierte Anzeige hochladen",passTitle:"Diese Anzeige entspricht den Unternehmensregeln",looked:"Vom Prüfer berücksichtigte Regeln:",humanH:"Von einer Person prüfen lassen",humanP:"Eine KI kann etwas übersehen. Öffnen Sie die Checkliste in einem neuen Tab, damit eine Person jeden Punkt bestätigt und abzeichnet.",openList:"Checkliste für die menschliche Prüfung öffnen",took:"Geprüft in {s} Sekunden.",reading:"Die Anzeige wird anhand der Regeln gelesen...",slow:"(dauert länger als üblich)",checking:"Wird geprüft...",checkBtn:"Anzeige prüfen",gen:"Korrigierte Anzeige erstellen",genning:"Korrigierte Anzeige wird geschrieben...",fixedH:"Korrigierte Anzeige",headline:"Überschrift",body:"Haupttext",cta:"Handlungsaufforderung",fine:"Kleingedrucktes",design:"Nötige Design-Änderungen",changes:"Was geändert wurde",copyAd:"Korrigierte Anzeige kopieren",dl:".txt herunterladen",regen:"Neu erstellen",aiNote:"Von einer KI anhand Ihrer Regeln geschrieben. Vor der Verwendung sollte eine Person sie prüfen.",rerun:"Die Sprache wurde geändert. Klicken Sie erneut auf „Anzeige prüfen“, um das Ergebnis in dieser Sprache zu erhalten.",problem:"Problem",fix:"Lösung",adLabel:"Anzeige",resLabel:"Ergebnis",img:"KI-generierte korrigierte Anzeige",imgWorking:"Bild der korrigierten Anzeige wird erzeugt ... das kann bis zu einer Minute dauern",imgDl:"Bild herunterladen",imgRegen:"Bild neu erstellen",imgNote:"KI-generiertes Bild. Prüfen Sie jedes Wort darin sorgfältig, bevor Sie es verwenden."},
    ru:{PASS:"ПРОЙДЕНО",FAIL:"НЕ ПРОЙДЕНО",failOne:"Нужно исправить 1 пункт, затем проверьте снова",failMany:"Нужно исправить: {n}, затем проверьте снова",failNone:"Это объявление нарушает правила компании",failSub:"На каждой карточке указаны правило, что не так в объявлении и как это исправить.",howFix:"Как исправить: ",defFix:"Измените объявление так, чтобы оно не нарушало это правило, и проверьте снова.",copyFix:"Скопировать список исправлений",copied:"Скопировано",again:"Загрузить исправленное объявление",passTitle:"Это объявление соответствует правилам компании",looked:"Правила, по которым шла проверка:",humanH:"Пусть проверит человек",humanP:"ИИ может что-то упустить. Откройте чек-лист в новой вкладке, чтобы человек подтвердил каждый пункт и поставил подпись.",openList:"Открыть чек-лист проверки человеком",took:"Проверено за {s} с.",reading:"Объявление сверяется с правилами...",slow:"(дольше обычного)",checking:"Проверка...",checkBtn:"Проверить объявление",gen:"Создать исправленное объявление",genning:"Пишется исправленное объявление...",fixedH:"Исправленное объявление",headline:"Заголовок",body:"Основной текст",cta:"Призыв к действию",fine:"Мелкий шрифт",design:"Необходимые изменения дизайна",changes:"Что изменено",copyAd:"Скопировать исправленное объявление",dl:"Скачать .txt",regen:"Создать заново",aiNote:"Написано ИИ на основе ваших правил. Перед использованием его должен проверить человек.",rerun:"Язык изменён. Нажмите «Проверить объявление» ещё раз, чтобы получить результат на этом языке.",problem:"Проблема",fix:"Исправление",adLabel:"Объявление",resLabel:"Результат",img:"Исправленное объявление, созданное ИИ",imgWorking:"Создаётся изображение исправленного объявления... это может занять до минуты",imgDl:"Скачать изображение",imgRegen:"Создать изображение заново",imgNote:"Изображение создано ИИ. Перед использованием внимательно проверьте каждое слово."}
  };
  var LANG=lsGet("bg-lang","en"), busy=false, last=null, state={provider:null,by:"AI",image:false};
  if(!I18N[LANG]) LANG="en";
  function t(k,v){
    var s=(I18N[LANG]&&I18N[LANG][k])||I18N.en[k]||k;
    if(v) Object.keys(v).forEach(function(n){ s=s.replace("{"+n+"}",v[n]); });
    return s;
  }
  var langSel=$("lang");
  langSel.value=LANG; document.documentElement.lang=LANG;
  function compliantLabel(v){ return LANG==="en"?(v==="PASS"?"Compliant":"Not Compliant"):t(v); }

  /* ---------- helpers ---------- */
  function ago(ts){
    var s=Math.max(0,Math.round((Date.now()-ts)/1000));
    if(s<60) return "just now";
    var m=Math.round(s/60); if(m<60) return m+" min ago";
    var hr=Math.round(m/60); if(hr<24) return hr+(hr===1?" hour ago":" hours ago");
    var d=Math.round(hr/24); return d+(d===1?" day ago":" days ago");
  }
  var SEVW={high:22,medium:12,low:8};
  function sevOf(v){ var s=String(v&&v.severity||"medium").toLowerCase(); return SEVW[s]?s:"medium"; }
  function scoreOf(rec){
    if(rec.verdict==="PASS") return 100;
    var p=0; (rec.violations||[]).forEach(function(v){ p+=SEVW[sevOf(v)]; });
    return Math.max(4,100-p);
  }
  function scoreColor(s){ return s>=80?"var(--pass)":s>=50?"var(--warn)":"var(--fail)"; }
  function hueOf(str){ var x=0; for(var i=0;i<str.length;i++) x=(x*31+str.charCodeAt(i))%360; return x; }
  function thumbEl(rec,size){
    if(rec.thumb){ var im=h("img",{class:"thumb",src:rec.thumb,alt:""}); return im; }
    var hu=rec.hue!=null?rec.hue:hueOf(rec.name||"x");
    var d=h("div",{class:"thumb"}); d.style.background="linear-gradient(135deg,hsl("+hu+",70%,52%),hsl("+((hu+50)%360)+",75%,38%))";
    d.innerHTML=svgOf(/\.pdf$/i.test(rec.file||"")?"file":"image"); return d;
  }
  function badgeEl(v){ return h("span",{class:"badge "+(v==="PASS"?"pass":"fail"),text:v==="PASS"?"Pass":"Fail"}); }
  function ringEl(score,size){
    var r=36,c=2*Math.PI*r,col=scoreColor(score);
    var w=h("div",{class:"ring"});
    w.innerHTML='<svg viewBox="0 0 84 84"><circle cx="42" cy="42" r="'+r+'" fill="none" stroke="currentColor" stroke-opacity=".14" stroke-width="8"/><circle class="arc" cx="42" cy="42" r="'+r+'" fill="none" stroke="'+col+'" stroke-width="8" stroke-linecap="round" style="stroke-dasharray:0 '+c.toFixed(1)+'"/></svg>';
    setTimeout(function(){ var a=w.querySelector(".arc"); if(a) a.style.strokeDasharray=(c*score/100).toFixed(1)+" "+c.toFixed(1); },90);
    w.appendChild(h("div",{class:"t"},[h("div",{},[String(score),h("small",{text:"/100"})])]));
    return w;
  }
  function sampleAdSvg(){
    return '<svg viewBox="0 0 320 150" preserveAspectRatio="xMidYMid slice" aria-label="Sample ad"><defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0f6b3a"/><stop offset="1" stop-color="#3fae5a"/></linearGradient></defs><rect width="320" height="150" fill="url(#sg)"/><ellipse cx="270" cy="30" rx="60" ry="26" fill="#7fd68a" opacity=".5" transform="rotate(-25 270 30)"/><ellipse cx="40" cy="140" rx="70" ry="22" fill="#7fd68a" opacity=".45" transform="rotate(20 40 140)"/><rect x="226" y="34" width="34" height="92" rx="8" fill="#e9fbe9" opacity=".95"/><rect x="233" y="22" width="20" height="16" rx="4" fill="#2f7d45"/><rect x="230" y="66" width="26" height="30" rx="5" fill="#5cc277"/><text x="22" y="62" font-family="Plus Jakarta Sans,Inter,sans-serif" font-weight="800" font-size="26" fill="#ffe27a">100% Natural</text><text x="22" y="90" font-family="Plus Jakarta Sans,Inter,sans-serif" font-weight="800" font-size="26" fill="#fff">Goodness</text><text x="22" y="116" font-family="Inter,sans-serif" font-size="10" fill="#e6ffe9">Pure Taste for a Healthier You</text></svg>';
  }

  /* ---------- data: real history + sample data ---------- */
  var DEMO_ON=lsGet("bg-demo","1")==="1";
  var real=[]; try{ real=JSON.parse(lsGet("bg-history","[]"))||[]; }catch(e){ real=[]; }
  function saveReal(){
    for(var guard=0; guard<60; guard++){
      if(lsSet("bg-history",JSON.stringify(real))) return;
      if(real.length<=1) return;
      real.pop();
    }
  }
  var VB={
    claim:{rule:"Unsupported claim",reason:'The ad claims "100% Natural" without approved evidence.',fix:"Remove the claim, or add approved proof and the substantiation line required by the claims policy.",severity:"high"},
    disc:{rule:"Missing mandatory disclaimer",reason:"Health related claims require a disclaimer, and none is shown.",fix:"Add the approved health disclaimer in the footer of the ad.",severity:"high"},
    brand:{rule:"Brand guideline violation",reason:"Logo size is smaller than the minimum requirement.",fix:"Enlarge the logo to at least the minimum size in the brand guide.",severity:"medium"},
    font:{rule:"Incorrect font usage",reason:"The headline uses a font outside the approved brand fonts.",fix:"Switch the headline to the approved brand typeface.",severity:"low"},
    guar:{rule:"Guaranteed results wording",reason:'The ad promises "guaranteed" results, which the policy prohibits.',fix:'Replace "guaranteed" with softer wording such as "designed to help".',severity:"high"},
    date:{rule:"Offer needs an end date",reason:"The discount is shown with no validity date.",fix:'Add a clear end date, for example "Valid until 31 Dec".',severity:"medium"},
    comp:{rule:"No competitor comparison",reason:"The ad names a competitor brand.",fix:"Remove the competitor name and any direct comparison.",severity:"high"},
    contrast:{rule:"Fine print is hard to read",reason:"The fine print has very low contrast against the background.",fix:"Use a darker text color or a lighter background for the fine print.",severity:"low"}
  };
  var CHK=["Claims & substantiation","Mandatory disclaimers","Logo usage","Brand fonts","Color palette"];
  var DEMO_BASE={n:248,p:216,i:32,s:12*248};
  var DEMO_REC=null;
  function demoRecords(){
    if(DEMO_REC) return DEMO_REC;
    var now=Date.now(), min=60000;
    function mk(id,name,file,mins,verdict,keys,tags,sec,sample){
      return {id:id,name:name,file:file,ts:now-mins*min,verdict:verdict,seconds:sec,demo:true,sample:!!sample,tags:tags,checked:CHK.slice(),
        violations:keys.map(function(k){ var o=VB[k]; return {rule:o.rule,reason:o.reason,fix:o.fix,severity:o.severity}; })};
    }
    DEMO_REC=[
      mk("d1","Product Launch Banner","banner_v2.png",1,"FAIL",["claim","disc","brand","font"],["Brand","Claims","Legal"],11,true),
      mk("d2","Summer Sale Banner","summer_sale.jpg",2,"PASS",[],["Brand","Offers"],9),
      mk("d3","Product Launch Poster","launch_poster.pdf",18,"FAIL",["guar","brand"],["Claims","Brand"],13),
      mk("d4","Instagram Post - New Flavour","new_flavour.png",60,"PASS",[],["Brand","Social"],8),
      mk("d5","Diwali Campaign Ad","diwali_ad.jpg",180,"PASS",[],["Brand","Offers"],10),
      mk("d6","Monsoon Offer Flyer","monsoon_flyer.pdf",420,"FAIL",["date","contrast"],["Offers","Legal"],14),
      mk("d7","Festive Combo Poster","festive_combo.png",1500,"PASS",[],["Brand","Offers"],9),
      mk("d8","Protein Bar Ad","protein_bar.jpg",2900,"FAIL",["claim","disc","comp"],["Claims","Legal"],15),
      mk("d9","Winter Collection Banner","winter_banner.png",4300,"PASS",[],["Brand"],8),
      mk("d10","Weekend Flash Deal","flash_deal.jpg",5800,"FAIL",["date","font"],["Offers","Brand"],12)
    ];
    return DEMO_REC;
  }
  function allRecs(){ return real.concat(DEMO_ON?demoRecords():[]).sort(function(a,b){ return b.ts-a.ts; }); }
  function findRec(id){ var a=allRecs(); for(var i=0;i<a.length;i++) if(a[i].id===id) return a[i]; return null; }
  function computeStats(){
    var b=DEMO_ON?DEMO_BASE:{n:0,p:0,i:0,s:0};
    var rp=0,ri=0,rs=0;
    real.forEach(function(r){ if(r.verdict==="PASS") rp++; ri+=(r.violations||[]).length; rs+=(r.seconds||0); });
    var n=b.n+real.length;
    return {n:n,rate:n?Math.round(100*(b.p+rp)/n):0,issues:b.i+ri,avg:n?Math.round((b.s+rs)/n):0};
  }
  function sparkEl(vals,col){
    var mx=Math.max.apply(null,vals), w=h("div",{class:"spark","aria-hidden":"true"});
    vals.forEach(function(v,k){ var i=h("i"); i.style.setProperty("--i",k); i.style.height=Math.max(4,Math.round(38*v/mx))+"px"; i.style.background=col; w.appendChild(i); });
    return w;
  }
  function countUp(el,txt){
    var m=/^(\d+)(.*)$/.exec(txt);
    if(!m||window.matchMedia("(prefers-reduced-motion:reduce)").matches){ el.textContent=txt; return; }
    var end=+m[1], suf=m[2], t0=null; el.textContent="0"+suf;
    function f(ts){ if(t0===null) t0=ts; var p=Math.min(1,(ts-t0)/900), e=1-Math.pow(1-p,3); el.textContent=Math.round(end*e)+suf; if(p<1) requestAnimationFrame(f); }
    requestAnimationFrame(f);
  }
  function statCards(into){
    var s=computeStats(), d=DEMO_ON;
    var defs=[
      {l:"Total Checks",v:String(s.n),d:d?"↑ 12%":"",up:1,sp:[8,12,9,14,11,16,13,18,15,20,17,22],c:"#6D5DF6"},
      {l:"Pass Rate",v:s.rate+"%",d:d?"↑ 5%":"",up:1,sp:[10,11,12,12,14,13,15,16,15,17,18,19],c:"#2ED8A3"},
      {l:"Issues Detected",v:String(s.issues),d:d?"↓ 18%":"",up:0,sp:[18,16,17,14,15,12,13,11,12,9,10,8],c:"#FF5F73"},
      {l:"Avg. Review Time",v:s.avg+"s",d:d?"↓ 40%":"",up:1,sp:[20,18,19,15,16,13,12,11,10,9,9,8],c:"#4C8DFF"}
    ];
    into.textContent="";
    defs.forEach(function(x){
      var num=h("span",{text:x.v});
      into.appendChild(h("div",{class:"card stat"},[h("small",{text:x.l}),h("b",{},[num,x.d?h("span",{class:"d "+(x.up?"up":"down"),text:x.d}):null]),sparkEl(x.sp,x.c)]));
      countUp(num,x.v);
    });
  }

  /* ---------- verdict views (home panel + report modal) ---------- */
  function issueRow(v){
    var s=sevOf(v);
    return h("div",{class:"iss"},[
      h("span",{class:"mk"+(s==="low"?" warn":"")},[icon(s==="low"?"warn":"x")]),
      h("div",{class:"tx"},[h("b",{text:v.rule||"-"}),h("span",{class:"r",text:v.reason||""})]),
      h("span",{class:"badge "+s,text:s.charAt(0).toUpperCase()+s.slice(1)})
    ]);
  }
  function matchedRows(rec){
    var v=(rec.violations||[]).map(function(x){ return String(x.rule||"").toLowerCase(); });
    return (rec.checked||[]).map(function(c){
      var lc=String(c).toLowerCase();
      var bad=v.some(function(r){ return r&&(r.indexOf(lc)>=0||lc.indexOf(r)>=0); });
      return h("div",{class:"iss"},[h("span",{class:"mk"+(bad?"":" ok")},[icon(bad?"x":"check")]),h("div",{class:"tx"},[h("b",{text:c}),h("span",{class:"r",text:bad?"Not met":"Matched"})])]);
    });
  }
  function suggestionRows(rec){
    var out=[];
    (rec.violations||[]).forEach(function(v){
      out.push(h("div",{class:"iss"},[h("span",{class:"mk ok"},[icon("edit")]),h("div",{class:"tx"},[h("b",{text:v.rule||"-"}),h("span",{class:"r",text:v.fix||t("defFix")})])]));
    });
    if(!out.length) out.push(h("div",{class:"iss"},[h("span",{class:"mk ok"},[icon("check")]),h("div",{class:"tx"},[h("b",{text:"No changes needed"}),h("span",{class:"r",text:"This ad follows every rule that was checked."})])]));
    return out;
  }
  function reviewHref(rec){
    var payload={ad:rec.file||rec.name,at:new Date(rec.ts).toISOString(),by:state.by,rules:rec.checked||[]};
    return "/review#d="+encodeURIComponent(JSON.stringify(payload));
  }
  function summaryText(rec){
    var n=(rec.violations||[]).length;
    if(rec.verdict==="PASS") return "This ad follows the company's policies. A person should still confirm it before it is published.";
    return "This ad breaks "+n+(n===1?" company policy":" company policies")+". Please review the issues below.";
  }
  function adImage(rec){
    var box=h("div",{class:"vimg"});
    if(rec.thumb) box.appendChild(h("img",{src:rec.thumb,alt:"The ad that was checked"}));
    else if(rec.sample) box.innerHTML=sampleAdSvg();
    else { var th=thumbEl(rec); th.style.width="100%"; th.style.height="100%"; th.style.borderRadius="0"; box.appendChild(th); }
    return box;
  }
  var panelTab="issues";
  function renderPanel(){
    var p=$("vpanel"); p.textContent="";
    var recs=allRecs(), rec=recs[0];
    p.appendChild(h("div",{class:"cardh"},[h("h3",{text:"AI Verdict",style:"margin:0"})]));
    if(!rec){
      p.appendChild(h("div",{class:"emptyv"},[icon("shield"),h("p",{text:"No checks yet. Upload an ad to see the AI verdict here."}),h("a",{class:"btnp",href:"#check",text:"Check an ad"})]));
      return;
    }
    var sc=scoreOf(rec);
    p.appendChild(h("div",{class:"vsum"},[
      h("div",{},[h("span",{class:"badge "+(rec.verdict==="PASS"?"pass":"fail"),text:compliantLabel(rec.verdict),style:"font-size:13px;padding:7px 14px"}),h("p",{text:summaryText(rec),style:"margin-top:10px;color:var(--muted);font-size:12.5px"})]),
      ringEl(sc)
    ]));
    p.appendChild(adImage(rec));
    var n=(rec.violations||[]).length;
    var tabs=h("div",{class:"tabs2",role:"tablist"});
    [["issues","Issues ("+n+")"],["matched","Matched Rules"],["sugg","Suggestions"]].forEach(function(x){
      tabs.appendChild(h("button",{type:"button",role:"tab","aria-selected":panelTab===x[0]?"true":"false",text:x[1],onclick:function(){ panelTab=x[0]; renderPanel(); }}));
    });
    p.appendChild(tabs);
    var box=h("div",{class:"vbox"});
    var rows=panelTab==="issues"?((rec.violations||[]).map(issueRow)):panelTab==="matched"?matchedRows(rec):suggestionRows(rec);
    if(panelTab==="issues"&&!rows.length) rows=[h("div",{class:"iss"},[h("span",{class:"mk ok"},[icon("check")]),h("div",{class:"tx"},[h("b",{text:"No issues found"}),h("span",{class:"r",text:"Every rule that was checked passed."})])])];
    rows.forEach(function(r){ box.appendChild(r); });
    p.appendChild(box);
    p.appendChild(h("div",{class:"vfoot"},[h("button",{class:"btn",type:"button",text:"View full report",onclick:function(){ openReport(rec); }})]));
  }
  function openReport(rec){
    var sc=scoreOf(rec), n=(rec.violations||[]).length;
    var kids=[
      h("div",{class:"mh"},[h("div",{},[h("h2",{text:rec.name,style:"font-size:22px"}),h("p",{class:"sub",text:(rec.file||"")+" • "+ago(rec.ts)+(rec.seconds?" • checked in "+rec.seconds+"s":""),style:"margin-top:4px"})])]),
      h("div",{class:"vsum"},[h("div",{},[h("span",{class:"badge "+(rec.verdict==="PASS"?"pass":"fail"),text:compliantLabel(rec.verdict),style:"font-size:13px;padding:7px 14px"}),h("p",{text:summaryText(rec),style:"margin-top:10px;color:var(--muted);font-size:13px"})]),ringEl(sc)]),
      adImage(rec)
    ];
    var secs=[["Issues ("+n+")",(rec.violations||[]).map(issueRow)],["Rules checked",matchedRows(rec)],["How to fix",rec.verdict==="FAIL"?suggestionRows(rec):[]]];
    secs.forEach(function(s){
      if(!s[1].length) return;
      var d=h("div",{class:"rep-sec"},[h("h4",{text:s[0]})]), g=h("div",{class:"vbox",style:"max-height:none"});
      s[1].forEach(function(r){ g.appendChild(r); }); d.appendChild(g); kids.push(d);
    });
    if(rec.verdict==="PASS"&&!rec.demo) kids.push(h("div",{class:"actions"},[h("a",{class:"btn green",href:reviewHref(rec),target:"_blank",rel:"noopener"},[t("openList"),icon("external")])]));
    if(rec.demo) kids.push(h("p",{class:"fine",text:"This is sample data shown for demonstration. Turn it off in Settings."}));
    openModal(h("div",{},kids),true);
  }

  /* ---------- rows ---------- */
  function recentRow(rec){
    var b=h("button",{type:"button",onclick:function(){ openReport(rec); }},[thumbEl(rec),h("div",{class:"grow"},[h("div",{class:"nm",text:rec.name}),h("small",{text:ago(rec.ts)})])]);
    return h("div",{class:"rrow"},[b,badgeEl(rec.verdict)]);
  }
  function tagsFor(rec){
    var tg=rec.tags||(rec.checked||[]).slice(0,2).map(function(c){ return c.length>16?c.slice(0,15)+"…":c; });
    var s=document.createDocumentFragment(); tg.slice(0,3).forEach(function(x){ s.appendChild(h("span",{class:"tag",text:x})); }); return s;
  }
  function tableRow(rec){
    var n=(rec.violations||[]).length;
    var vr=h("div",{class:"vr"});
    var btn=h("button",{type:"button","aria-haspopup":"true"},["View Report",icon("chev")]);
    btn.addEventListener("click",function(e){
      e.stopPropagation();
      var had=vr.querySelector(".menu"); closePops(); if(had) return;
      var m=h("div",{class:"menu",style:"right:0;left:auto"});
      m.appendChild(h("button",{type:"button",onclick:function(){ closePops(); openReport(rec); }},[icon("file"),"Open report"]));
      if(rec.verdict==="PASS"&&!rec.demo) m.appendChild(h("button",{type:"button",onclick:function(){ closePops(); window.open(reviewHref(rec),"_blank","noopener"); }},[icon("person"),"Human review checklist"]));
      vr.appendChild(m);
    });
    vr.appendChild(btn);
    return h("div",{class:"trow"},[
      h("div",{class:"who"},[thumbEl(rec),h("div",{},[h("b",{text:rec.name}),h("small",{text:(rec.file||"")+" • "+ago(rec.ts)})])]),
      h("div",{},[badgeEl(rec.verdict)]),
      h("div",{class:"issuesfound"+(n?"":" ok")},[icon(n?"warn":"check"),n?(n+(n===1?" issue found":" issues found")):"No issues"]),
      h("div",{},[tagsFor(rec)]),
      h("div",{},[vr])
    ]);
  }
  function tableHead(){
    return h("div",{class:"trow head"},[h("div",{text:"Ad"}),h("div",{text:"Status"}),h("div",{text:"Issues"}),h("div",{text:"Policy matched"}),h("div",{text:"Actions"})]);
  }
  function emptyTable(into,msg){
    into.appendChild(h("div",{class:"emptyv"},[icon("clock"),h("p",{text:msg}),h("a",{class:"btnp",href:"#check",text:"Check an ad"})]));
  }

  /* ---------- pages ---------- */
  function renderHome(){
    var recs=allRecs(), rl=$("recentList"); rl.textContent="";
    if(!recs.length) rl.appendChild(h("p",{class:"sub",text:"Your checks will appear here."}));
    recs.slice(0,4).forEach(function(r){ rl.appendChild(recentRow(r)); });
    statCards($("stats"));
    var lt=$("latestTbl"); lt.textContent="";
    if(!recs.length) emptyTable(lt,"No verifications yet.");
    else { lt.appendChild(tableHead()); recs.slice(0,3).forEach(function(r){ lt.appendChild(tableRow(r)); }); }
    renderPanel();
  }
  var hFilter="all";
  function renderHistory(){
    var q=$("hq").value.trim().toLowerCase();
    var recs=allRecs().filter(function(r){ return (hFilter==="all"||r.verdict===hFilter)&&(!q||(r.name+" "+(r.file||"")).toLowerCase().indexOf(q)>=0); });
    var tb=$("histTbl"); tb.textContent="";
    if(!recs.length){ emptyTable(tb,"No verifications match."); return; }
    tb.appendChild(tableHead()); recs.forEach(function(r){ tb.appendChild(tableRow(r)); });
  }
  $("hq").addEventListener("input",renderHistory);
  Array.prototype.forEach.call($("hseg").querySelectorAll("button"),function(b){
    b.addEventListener("click",function(){
      hFilter=b.getAttribute("data-f");
      Array.prototype.forEach.call($("hseg").querySelectorAll("button"),function(x){ x.setAttribute("aria-pressed",x===b?"true":"false"); });
      renderHistory();
    });
  });
  function csvCell(s){ s=String(s==null?"":s); return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s; }
  $("csvBtn").addEventListener("click",function(){
    var rows=[["Ad","File","Result","Score","Issues","Checked at"]];
    allRecs().forEach(function(r){ rows.push([r.name,r.file||"",r.verdict,scoreOf(r),(r.violations||[]).length,new Date(r.ts).toISOString()]); });
    var blob=new Blob([rows.map(function(r){ return r.map(csvCell).join(","); }).join("\n")],{type:"text/csv;charset=utf-8"});
    var a=h("a",{href:URL.createObjectURL(blob),download:"verifications.csv"}); document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function(){ URL.revokeObjectURL(a.href); },1000);
  });
  function barRow(label,val,max){
    var f=h("div",{class:"fill"}); f.style.width=(max?Math.round(100*val/max):0)+"%";
    return h("div",{class:"bar"},[h("span",{class:"nm",text:label,title:label}),h("div",{class:"track"},[f]),h("b",{text:String(val)})]);
  }
  function renderReports(){
    statCards($("repStats"));
    var recs=allRecs(), pass=recs.filter(function(r){return r.verdict==="PASS";}).length, fail=recs.length-pass;
    var d=$("repDonut"); d.textContent="";
    var tot=recs.length||1, c=2*Math.PI*52;
    var w=h("div",{class:"ring",style:"width:130px;height:130px"});
    w.innerHTML='<svg viewBox="0 0 130 130"><circle cx="65" cy="65" r="52" fill="none" stroke="var(--fail)" stroke-width="16"/><circle cx="65" cy="65" r="52" fill="none" stroke="var(--pass)" stroke-width="16" stroke-dasharray="'+(c*pass/tot).toFixed(1)+' '+c.toFixed(1)+'"/></svg>';
    w.appendChild(h("div",{class:"t"},[h("div",{},[String(recs.length),h("small",{text:"checks"})])]));
    d.appendChild(w);
    d.appendChild(h("div",{class:"legend"},[h("span",{},[h("i",{style:"background:var(--pass)"}),"Pass: "+pass]),h("span",{},[h("i",{style:"background:var(--fail)"}),"Fail: "+fail])]));
    var wk=$("repWeek"); wk.textContent="";
    var days=[],names=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
    for(var i=6;i>=0;i--){ var dt=new Date(); dt.setHours(0,0,0,0); dt.setDate(dt.getDate()-i); days.push({s:dt.getTime(),e:dt.getTime()+864e5,l:names[dt.getDay()],n:0}); }
    recs.forEach(function(r){ days.forEach(function(x){ if(r.ts>=x.s&&r.ts<x.e) x.n++; }); });
    var mx=Math.max(1,Math.max.apply(null,days.map(function(x){return x.n;})));
    days.forEach(function(x){ var i=h("i"); i.style.height=Math.max(4,Math.round(110*x.n/mx))+"px"; wk.appendChild(h("div",{},[h("em",{text:String(x.n)}),i,h("span",{text:x.l})])); });
    var sev={high:0,medium:0,low:0}, top={};
    recs.forEach(function(r){ (r.violations||[]).forEach(function(v){ sev[sevOf(v)]++; var k=v.rule||"-"; top[k]=(top[k]||0)+1; }); });
    var sv=$("repSev"); sv.textContent="";
    var smx=Math.max(1,sev.high,sev.medium,sev.low);
    [["High","high"],["Medium","medium"],["Low","low"]].forEach(function(x){ sv.appendChild(barRow(x[0],sev[x[1]],smx)); });
    var tp=$("repTop"); tp.textContent="";
    var arr=Object.keys(top).map(function(k){return [k,top[k]];}).sort(function(a,b){return b[1]-a[1];}).slice(0,5);
    if(!arr.length) tp.appendChild(h("p",{class:"sub",text:"No broken rules yet."}));
    arr.forEach(function(x){ tp.appendChild(barRow(x[0],x[1],arr[0][1])); });
  }
  var INTEG=[["Slack","Get a message in a channel when an ad fails.","#4A154B"],["Google Drive","Check ads straight from a shared folder.","#1A73E8"],["Figma","Review designs before they leave the studio.","#F24E1E"],["Canva","Check a design without exporting it first.","#00A3B4"],["Gmail","Email verdicts to your team automatically.","#D93025"],["Webhooks & API","Send ads from your own tools and get results back.","#6D5DF6"]];
  function renderInteg(){
    var g=$("integGrid"); g.textContent="";
    INTEG.forEach(function(x){
      var lg=h("div",{class:"lg",text:x[0].charAt(0)}); lg.style.background=x[2];
      g.appendChild(h("div",{class:"card"},[h("div",{class:"top2"},[lg,h("span",{class:"badge soon",text:"Coming soon"})]),h("h3",{text:x[0],style:"margin:0"}),h("p",{text:x[1]}),h("button",{class:"btn",type:"button",disabled:"disabled",text:"Connect"})]));
    });
  }
  function renderBell(){
    var pop=$("bellPop"); pop.textContent="";
    pop.appendChild(h("div",{class:"ph",text:"Notifications"}));
    var recs=allRecs().slice(0,5);
    if(!recs.length) pop.appendChild(h("div",{class:"none",text:"You're all caught up."}));
    recs.forEach(function(r){
      var n=(r.violations||[]).length;
      pop.appendChild(h("button",{class:"it",type:"button",onclick:function(){ closePops(); openReport(r); }},[thumbEl(r),h("div",{},[h("b",{text:r.name+(r.verdict==="PASS"?" passed":" failed"),style:"font-size:13px"}),h("small",{text:(n?n+(n===1?" issue found • ":" issues found • "):"")+ago(r.ts)})])]));
    });
  }
  function updateDot(){
    var seen=Number(lsGet("bg-seen","0"));
    $("bellDot").hidden=!real.some(function(r){ return r.ts>seen; });
  }
  $("bellBtn").addEventListener("click",function(e){
    e.stopPropagation();
    var pop=$("bellPop"), open=pop.hidden; closePops();
    if(open){ renderBell(); pop.hidden=false; lsSet("bg-seen",String(Date.now())); updateDot(); }
  });
  var qt=null;
  $("q").addEventListener("input",function(){
    var q=this.value.trim().toLowerCase(), box=$("qres"); box.textContent="";
    if(!q){ box.hidden=true; return; }
    var pages=[["Home","#home"],["Check Ad","#check"],["Past Verifications","#history"],["Company Rules","#rules"],["Team","#team"],["Reports","#reports"],["Integrations","#integrations"],["Settings","#settings"],["How it works","#how"]]
      .filter(function(p){ return p[0].toLowerCase().indexOf(q)>=0; });
    var recs=allRecs().filter(function(r){ return (r.name+" "+(r.file||"")+" "+(r.violations||[]).map(function(v){return v.rule;}).join(" ")).toLowerCase().indexOf(q)>=0; }).slice(0,6);
    if(recs.length){ box.appendChild(h("div",{class:"ph",text:"Verifications"})); recs.forEach(function(r){ box.appendChild(h("button",{class:"it",type:"button",onclick:function(){ closePops(); openReport(r); }},[thumbEl(r),h("div",{},[h("b",{text:r.name,style:"font-size:13px"}),h("small",{text:r.verdict==="PASS"?"Pass":"Fail"})])])); }); }
    if(pages.length){ box.appendChild(h("div",{class:"ph",text:"Pages"})); pages.forEach(function(p){ box.appendChild(h("button",{class:"it",type:"button",onclick:function(){ closePops(); $("q").value=""; location.hash=p[1]; }},[icon("arrow"),h("b",{text:p[0],style:"font-size:13px"})])); }); }
    if(!recs.length&&!pages.length) box.appendChild(h("div",{class:"none",text:"Nothing found."}));
    box.hidden=false;
  });
  $("q").addEventListener("focus",function(){ if(this.value.trim()) this.dispatchEvent(new Event("input")); });

  /* ---------- routing ---------- */
  var PAGES=["home","check","history","rules","team","reports","integrations","settings","how"];
  function route(){
    var n=(location.hash||"#home").slice(1);
    if(PAGES.indexOf(n)<0) n="home";
    PAGES.forEach(function(p){
      $("p-"+p).hidden=p!==n;
      var a=$("nav-"+p); if(a){ a.classList.toggle("on",p===n); if(p===n) a.setAttribute("aria-current","page"); else a.removeAttribute("aria-current"); }
    });
    closePops();
    if(n==="home") renderHome();
    if(n==="history") renderHistory();
    if(n==="reports") renderReports();
    if(n==="integrations") renderInteg();
    if(n==="check"||n==="rules") refresh();
    Array.prototype.forEach.call($("p-"+n).querySelectorAll(".card"),function(c,i){ c.style.animationDelay=(i*70)+"ms"; });
    window.scrollTo(0,0);
  }
  window.addEventListener("hashchange",route);

  /* ---------- status ---------- */
  fetch("/api/status").then(function(r){return r.json()}).then(function(s){
    state.provider=s.provider; state.by=s.by; state.image=!!s.image;
    $("appname").textContent=s.name; document.title=s.name;
    var pi=$("provInfo");
    pi.textContent=(s.provider==="gemini"?"Google Gemini":s.provider==="claude"?"Anthropic Claude":"No API key set")+
      (s.provider?(s.image?". Corrected ad images are available.":". Corrected ad images need a Gemini key."):". Set GEMINI_API_KEY or ANTHROPIC_API_KEY and restart.");
    if(!s.provider){
      var b=$("banner"); b.hidden=false;
      b.textContent="No API key is set, so checks will fail. Set GEMINI_API_KEY (free at aistudio.google.com) or ANTHROPIC_API_KEY, then restart the app.";
    }
  }).catch(function(){ $("provInfo").textContent="Could not reach the server."; });

  /* ---------- inputs: ad + rules ---------- */
  var policyFile=null, adFile=null, previewUrl=null;
  var go=$("go"), result=$("result"), sheet=$("sheet"), stage=$("stage");
  var MAXB=20*1024*1024, SAMPLE_RULES="1. Do not claim guaranteed or 100% results.\n2. Health-related claims need an approved disclaimer.\n3. The company logo must appear at the minimum size.\n4. Use only approved brand fonts and colors.\n5. Offers must show a clear end date.\n6. Do not name or compare with competitors.";
  function hasPolicy(){ return !!policyFile || $("policyText").value.trim().length>0; }
  function refresh(){
    go.disabled=!(hasPolicy() && adFile) || busy;
    var rs=$("rulesStatus"); rs.textContent="";
    var ok=hasPolicy();
    rs.className="card rulesline"+(ok?"":" warn");
    rs.appendChild(h("span",{class:"ri"},[icon(ok?"rules":"warn")]));
    var tx=$("policyText").value.trim().length;
    rs.appendChild(h("div",{class:"rt"},[h("b",{text:ok?"Company rules ready":"No company rules yet"}),h("small",{text:ok?(policyFile?("From file: "+policyFile.name):(tx+" characters of rules saved")):"Add the rules your ad has to follow."})]));
    if(ok) rs.appendChild(h("a",{class:"btn small",href:"#rules",text:"Edit rules"}));
    else {
      rs.appendChild(h("button",{class:"btn small",type:"button",text:"Use sample rules",onclick:useSample}));
      rs.appendChild(h("a",{class:"btnp small",href:"#rules",text:"Add rules",style:"min-height:34px;padding:0 14px;font-size:13px"}));
    }
  }
  function useSample(){ $("policyText").value=SAMPLE_RULES; saveRulesText(); refresh(); toast("Sample rules added. You can edit them under Company Rules."); }
  function emptyState(){
    sheet.className="card sheet";
    var svg='<svg viewBox="0 0 120 90" aria-hidden="true"><rect x="22" y="8" width="62" height="76" rx="8" fill="none" stroke="currentColor" stroke-opacity=".5" stroke-width="2" stroke-dasharray="5 4"/><rect x="32" y="22" width="30" height="5" rx="2.5" fill="currentColor" opacity=".35"/><rect x="32" y="34" width="42" height="5" rx="2.5" fill="currentColor" opacity=".25"/><rect x="32" y="46" width="36" height="5" rx="2.5" fill="currentColor" opacity=".25"/><circle cx="88" cy="62" r="18" fill="#6D5DF6"/><path d="M80 62l6 6 11-12" fill="none" stroke="#FFC247" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    var d=h("div",{class:"empty"},[h("p",{text:"Add your ad and make sure the company rules are set, then press Check ad. Your result appears here."})]);
    d.insertAdjacentHTML("afterbegin",svg);
    show(d);
  }
  function show(node){ result.textContent=""; result.appendChild(node); }
  function bindDrop(dropId,inputId,onPick){
    var drop=$(dropId), input=$(inputId);
    input.addEventListener("click",function(){ input.value=""; });
    input.addEventListener("change",function(){ if(input.files[0]) onPick(input.files[0]); });
    ["dragenter","dragover"].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();drop.classList.add("over")})});
    ["dragleave","drop"].forEach(function(ev){drop.addEventListener(ev,function(e){e.preventDefault();drop.classList.remove("over")})});
    drop.addEventListener("drop",function(e){ if(e.dataTransfer.files[0]) onPick(e.dataTransfer.files[0]); });
  }
  function okType(f){ return /^(image\/(png|jpe?g|gif|webp)|application\/pdf)$/.test(f.type)||/\.(png|jpe?g|gif|webp|pdf)$/i.test(f.name); }

  /* the ad */
  function setAd(f){
    if(!okType(f)){ toast("Only PNG, JPG, GIF, WEBP or PDF files can be checked."); return false; }
    if(f.size>MAXB){ toast("That file is too large. The limit is 20 MB."); return false; }
    adFile=f; $("adName").textContent=f.name; $("adDrop").classList.add("has-file"); $("adClear").hidden=false;
    stage.hidden=false; stage.textContent="";
    if(previewUrl){ URL.revokeObjectURL(previewUrl); previewUrl=null; }
    if(f.type.indexOf("image/")===0){ previewUrl=URL.createObjectURL(f); stage.appendChild(h("img",{src:previewUrl,alt:"Preview of the ad"})); }
    else stage.appendChild(h("div",{class:"pdfcard"},[icon("file"),h("div",{},[h("strong",{text:f.name}),h("span",{text:"PDF document"})])]));
    last=null; emptyState(); refresh(); return true;
  }
  bindDrop("adDrop","adFile",setAd);
  bindDrop("homeDrop","homeFile",function(f){ if(setAd(f)) location.hash="#check"; });
  $("adClear").addEventListener("click",function(){
    adFile=null; $("adFile").value=""; $("adName").textContent="Drop the ad here, or click to choose";
    $("adDrop").classList.remove("has-file"); $("adClear").hidden=true; stage.hidden=true; stage.textContent=""; refresh();
  });
  $("homeUp").addEventListener("click",function(){ $("homeFile").click(); });
  $("homeCaret").addEventListener("click",function(e){ e.stopPropagation(); var m=$("upMenu"), open=m.hidden; closePops(); m.hidden=!open; });
  $("mDevice").addEventListener("click",function(){ closePops(); $("homeFile").click(); });
  function urlSoon(){ closePops(); toast("Importing an ad from a URL is coming soon. For now, upload the file."); }
  $("mUrl").addEventListener("click",urlSoon);
  $("urlGo").addEventListener("click",urlSoon);
  $("urlIn").addEventListener("keydown",function(e){ if(e.key==="Enter") urlSoon(); });

  /* the rules (saved in this browser) */
  var saveT=null;
  function flashSaved(msg){ var s=$("savedTag"); s.textContent=msg; clearTimeout(saveT); saveT=setTimeout(function(){ s.textContent=""; },2200); }
  function saveRulesText(){ lsSet("bg-rules-text",$("policyText").value); }
  function readAsDataUrl(f){ return new Promise(function(res,rej){ var r=new FileReader(); r.onload=function(){res(r.result);}; r.onerror=rej; r.readAsDataURL(f); }); }
  function dataUrlToFile(o){
    var parts=o.data.split(","), bin=atob(parts[1]), arr=new Uint8Array(bin.length);
    for(var i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);
    return new File([arr],o.name,{type:o.type});
  }
  function showPolicyFile(f){
    policyFile=f; $("policyName").textContent=f?f.name:"Drop the rules file here, or click to choose";
    $("policyDrop").classList.toggle("has-file",!!f); $("policyClear").hidden=!f;
  }
  bindDrop("policyDrop","policyFile",function(f){
    if(f.size>MAXB){ toast("That file is too large. The limit is 20 MB."); return; }
    showPolicyFile(f); last=null; refresh();
    if(f.size<=1500000){
      readAsDataUrl(f).then(function(d){ if(lsSet("bg-rules-file",JSON.stringify({name:f.name,type:f.type,data:d}))) flashSaved("Saved in this browser."); else flashSaved("Using this file for now (too big to save)."); }).catch(function(){});
    } else flashSaved("Using this file for now (too big to save in the browser).");
  });
  $("policyClear").addEventListener("click",function(){ showPolicyFile(null); $("policyFile").value=""; lsDel("bg-rules-file"); refresh(); });
  var typeT=null;
  $("policyText").addEventListener("input",function(){ refresh(); clearTimeout(typeT); typeT=setTimeout(function(){ saveRulesText(); flashSaved("Saved in this browser."); },400); });
  $("sampleRules").addEventListener("click",function(){ $("policyText").value=SAMPLE_RULES; saveRulesText(); refresh(); flashSaved("Sample rules added and saved."); });
  (function restoreRules(){
    $("policyText").value=lsGet("bg-rules-text","");
    try{ var o=JSON.parse(lsGet("bg-rules-file","null")); if(o&&o.data) showPolicyFile(dataUrlToFile(o)); }catch(e){}
  })();

  /* ---------- result helpers ---------- */
  function stamp(v){
    var w=t(v);
    return h("div",{class:"stampwrap"},[h("span",{class:"stamp "+(v==="PASS"?"pass":"fail")+" slam"+(w.length>8?" long":""),text:w})]);
  }
  function fixText(d){
    var lines=[t("adLabel")+": "+adFile.name,t("resLabel")+": "+t("FAIL"),""];
    d.violations.forEach(function(v,i){
      lines.push((i+1)+". "+v.rule); lines.push("   "+t("problem")+": "+v.reason); lines.push("   "+t("fix")+": "+(v.fix||t("defFix"))); lines.push("");
    });
    return lines.join("\n");
  }
  function copy(text,btn,done){
    var orig=btn.textContent;
    function ok(){ btn.textContent=done; setTimeout(function(){btn.textContent=orig;},1800); }
    function fallback(){
      var ta=h("textarea",{style:"position:fixed;opacity:0"}); ta.value=text; document.body.appendChild(ta); ta.select();
      try{document.execCommand("copy"); ok();}catch(e){btn.textContent="Copy failed";} document.body.removeChild(ta);
    }
    if(navigator.clipboard&&navigator.clipboard.writeText){ navigator.clipboard.writeText(text).then(ok,fallback); } else fallback();
  }
  function took(d){ return d.seconds?h("p",{class:"fine",text:t("took",{s:d.seconds})}):null; }

  /* ---------- corrected ad (generative AI) ---------- */
  function fixAdText(f){
    var L=[];
    if(f.headline) L.push(f.headline,""); if(f.body) L.push(f.body,"");
    if(f.call_to_action) L.push(f.call_to_action,""); if(f.fine_print) L.push(f.fine_print,"");
    return L.join("\n").trim();
  }
  function fullFixText(f){
    var L=[fixAdText(f)];
    if(f.design_changes.length){ L.push("","== "+t("design")+" =="); f.design_changes.forEach(function(x){L.push("- "+x);}); }
    if(f.changes.length){ L.push("","== "+t("changes")+" =="); f.changes.forEach(function(x){L.push("- "+x);}); }
    return L.join("\n");
  }
  function renderFix(f,out){
    var card=h("div",{class:"fixed"});
    card.appendChild(h("h3",{class:"fixed-h",text:t("fixedH")}));
    var ad=h("div",{class:"adcard"});
    if(f.headline) ad.appendChild(h("h4",{class:"ad-h",text:f.headline}));
    if(f.body) ad.appendChild(h("p",{class:"ad-b",text:f.body}));
    if(f.call_to_action) ad.appendChild(h("span",{class:"ad-cta",text:f.call_to_action}));
    if(f.fine_print) ad.appendChild(h("p",{class:"ad-fine",text:f.fine_print}));
    card.appendChild(ad);
    var imgBox=h("div",{class:"imgbox"}); card.appendChild(imgBox);
    function list(title,items){
      if(!items.length) return;
      var ul=h("ul",{class:"fixlist"}); items.forEach(function(x){ ul.appendChild(h("li",{text:x})); });
      card.appendChild(h("h4",{class:"fl-h",text:title})); card.appendChild(ul);
    }
    list(t("design"),f.design_changes); list(t("changes"),f.changes);
    var cp=h("button",{class:"btn",type:"button",text:t("copyAd")});
    cp.addEventListener("click",function(){ copy(fullFixText(f),cp,t("copied")); });
    var dl=h("button",{class:"btn",type:"button"},[icon("download"),t("dl")]);
    dl.addEventListener("click",function(){
      var blob=new Blob([fullFixText(f)],{type:"text/plain;charset=utf-8"});
      var a=h("a",{href:URL.createObjectURL(blob),download:adFile.name.replace(/\.[^.]+$/,"")+"-corrected.txt"});
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(function(){URL.revokeObjectURL(a.href);},1000);
    });
    card.appendChild(h("div",{class:"actions"},[cp,dl]));
    card.appendChild(h("p",{class:"fine",text:t("aiNote")}));
    out.appendChild(card);
    card.scrollIntoView({block:"nearest",behavior:"smooth"});
    return imgBox;
  }
  function generateImage(f,d,box){
    var fd=new FormData();
    fd.append("ad",adFile); fd.append("fix",JSON.stringify(f)); fd.append("violations",JSON.stringify(d.violations));
    box.textContent="";
    box.appendChild(h("div",{class:"working"},[h("span",{class:"spin"}),h("p",{text:t("imgWorking")})]));
    var ac=new AbortController(), killer=setTimeout(function(){ac.abort();},200000);
    function retry(){ return h("button",{class:"btn",type:"button",text:t("imgRegen"),onclick:function(){generateImage(f,d,box);}}); }
    fetch("/api/fix-image",{method:"POST",body:fd,signal:ac.signal}).then(function(r){
      return r.json().then(function(j){return {ok:r.ok,body:j};});
    }).then(function(res){
      box.textContent="";
      if(!res.ok){ box.appendChild(h("p",{class:"err",text:res.body.error||"Something went wrong. Try again."})); box.appendChild(retry()); return; }
      var ext=(res.body.mime||"image/png").indexOf("jpeg")>-1?"jpg":"png";
      box.appendChild(h("h4",{class:"fl-h",text:t("img")}));
      box.appendChild(h("img",{src:res.body.image,alt:t("img")}));
      var dl=h("a",{class:"btn",href:res.body.image,download:adFile.name.replace(/\.[^.]+$/,"")+"-corrected."+ext},[icon("download"),t("imgDl")]);
      box.appendChild(h("div",{class:"actions"},[dl,retry()]));
      box.appendChild(h("p",{class:"fine",text:t("imgNote")}));
    }).catch(function(e){
      box.textContent="";
      box.appendChild(h("p",{class:"err",text:(e&&e.name==="AbortError")?"The image is taking too long. Try again.":"Could not reach the app. Check that it is still running, then try again."}));
      box.appendChild(retry());
    }).then(function(){ clearTimeout(killer); });
  }
  function policyForm(fd){ if(policyFile) fd.append("policy",policyFile); else fd.append("policy_text",$("policyText").value); }
  function generateFix(d,btn,lab,out){
    var fd=new FormData();
    fd.append("ad",adFile); policyForm(fd);
    fd.append("violations",JSON.stringify(d.violations)); fd.append("lang",LANG);
    btn.disabled=true; out.textContent="";
    out.appendChild(h("div",{class:"working"},[h("span",{class:"spin"}),h("p",{text:t("genning")})]));
    var ac=new AbortController(), killer=setTimeout(function(){ac.abort();},150000);
    fetch("/api/fix",{method:"POST",body:fd,signal:ac.signal}).then(function(r){
      return r.json().then(function(j){return {ok:r.ok,body:j};});
    }).then(function(res){
      out.textContent="";
      if(!res.ok){ out.appendChild(h("p",{class:"err",text:res.body.error||"Something went wrong. Try again."})); return; }
      var imgBox=renderFix(res.body,out); lab.textContent=t("regen"); if(state.image) generateImage(res.body,d,imgBox);
    }).catch(function(e){
      out.textContent="";
      out.appendChild(h("p",{class:"err",text:(e&&e.name==="AbortError")?"This is taking too long. Try again.":"Could not reach the app. Check that it is still running, then try again."}));
    }).then(function(){ clearTimeout(killer); btn.disabled=false; });
  }

  function renderFail(d,note){
    last={kind:"fail",d:d};
    sheet.className="card sheet fail";
    var box=h("div"), n=d.violations.length;
    box.appendChild(stamp("FAIL"));
    box.appendChild(h("h3",{class:"vtitle",text:n?(n===1?t("failOne"):t("failMany",{n:n})):t("failNone")}));
    box.appendChild(h("p",{class:"vsub",text:t("failSub")}));
    var ul=h("ul",{class:"vlist"});
    d.violations.forEach(function(v){
      var fix=v.fix||t("defFix");
      ul.appendChild(h("li",{class:"vcard"},[h("h4",{text:v.rule||"-"}),h("p",{class:"wrong",text:v.reason}),h("div",{class:"fixbox"},[icon("edit"),h("div",{},[h("strong",{text:t("howFix")}),fix])])]));
    });
    box.appendChild(ul);
    if(n){
      var lab=h("span",{text:t("gen")});
      var genBtn=h("button",{class:"btn gen",type:"button"},[icon("sparkle"),lab]);
      var out=h("div",{class:"fixout"});
      genBtn.addEventListener("click",function(){ generateFix(d,genBtn,lab,out); });
      box.appendChild(genBtn); box.appendChild(out);
    }
    var copyBtn=h("button",{class:"btn",type:"button",text:t("copyFix")});
    copyBtn.addEventListener("click",function(){copy(fixText(d),copyBtn,t("copied"));});
    var again=h("button",{class:"btnp",type:"button"},[icon("upload"),t("again")]);
    again.addEventListener("click",function(){ $("adFile").click(); });
    box.appendChild(h("div",{class:"actions"},[again,copyBtn]));
    var t1=took(d); if(t1) box.appendChild(t1);
    if(note) box.appendChild(note);
    show(box);
  }
  function renderPass(d,note){
    last={kind:"pass",d:d};
    sheet.className="card sheet pass";
    var box=h("div");
    box.appendChild(stamp("PASS"));
    box.appendChild(h("h3",{class:"vtitle",text:t("passTitle")}));
    if(d.checked&&d.checked.length){
      box.appendChild(h("p",{class:"vsub",text:t("looked")}));
      var ul=h("ul",{class:"checked"});
      d.checked.forEach(function(r){ ul.appendChild(h("li",{},[icon("check"),h("span",{text:r})])); });
      box.appendChild(ul);
    }
    var payload={ad:adFile.name,at:new Date().toISOString(),by:state.by,rules:d.checked||[]};
    var href="/review#d="+encodeURIComponent(JSON.stringify(payload));
    var link=h("a",{class:"btn green",href:href,target:"_blank",rel:"noopener"},[t("openList"),icon("external")]);
    box.appendChild(h("div",{class:"human"},[
      h("div",{class:"row"},[icon("person"),h("div",{},[h("h4",{text:t("humanH"),style:"font-size:16px;margin-bottom:4px"}),h("p",{text:t("humanP")})])]),
      link
    ]));
    var t2=took(d); if(t2) box.appendChild(t2);
    if(note) box.appendChild(note);
    show(box);
    if(!note && window.__confetti) window.__confetti();
  }

  /* ---------- run a check ---------- */
  function makeThumb(f){
    return new Promise(function(res){
      if(!f||f.type.indexOf("image/")!==0) return res(null);
      var u=URL.createObjectURL(f), im=new Image();
      im.onload=function(){
        try{
          var w=360, s=Math.min(1,w/im.width), c=document.createElement("canvas");
          c.width=Math.max(1,Math.round(im.width*s)); c.height=Math.max(1,Math.round(im.height*s));
          c.getContext("2d").drawImage(im,0,0,c.width,c.height);
          res(c.toDataURL("image/jpeg",.72));
        }catch(e){ res(null); }
        URL.revokeObjectURL(u);
      };
      im.onerror=function(){ URL.revokeObjectURL(u); res(null); };
      im.src=u;
    });
  }
  function saveRecord(d,file){
    var rec={id:"r"+Date.now(),name:file.name.replace(/\.[^.]+$/,"").replace(/[_]+/g," "),file:file.name,ts:Date.now(),
      verdict:d.verdict,seconds:d.seconds||0,checked:d.checked||[],thumb:null,demo:false,
      violations:(d.violations||[]).map(function(v){ return {rule:v.rule,reason:v.reason,fix:v.fix,severity:v.severity||"medium"}; })};
    real.unshift(rec); if(real.length>40) real.length=40;
    saveReal(); updateDot();
    makeThumb(file).then(function(th){ if(th){ rec.thumb=th; saveReal(); } });
  }
  go.addEventListener("click",function(){
    var fd=new FormData(), sentAd=adFile;
    fd.append("ad",adFile); policyForm(fd); fd.append("lang",LANG);
    busy=true; last=null;
    go.disabled=true; go.textContent=t("checking");
    sheet.className="card sheet"; stage.classList.add("scanning");
    var t0=Date.now(), msg=h("p",{text:t("reading")});
    show(h("div",{class:"working"},[h("span",{class:"spin"}),msg]));
    var timer=setInterval(function(){
      var sec=Math.round((Date.now()-t0)/1000);
      msg.textContent=t("reading")+" "+sec+"s"+(sec>25?" "+t("slow"):"");
    },1000);
    var ac=new AbortController(), killer=setTimeout(function(){ac.abort();},150000);
    fetch("/api/check",{method:"POST",body:fd,signal:ac.signal}).then(function(r){
      return r.json().then(function(j){return {ok:r.ok,body:j};});
    }).then(function(res){
      if(!res.ok){ show(h("p",{class:"err",text:res.body.error||"Something went wrong. Try again."})); return; }
      saveRecord(res.body,sentAd);
      if(res.body.verdict==="PASS") renderPass(res.body); else renderFail(res.body);
    }).catch(function(e){
      show(h("p",{class:"err",text:(e&&e.name==="AbortError")
        ? "This is taking too long. Check your internet connection and try again."
        : "Could not reach the app. Check that it is still running, then try again."}));
    }).then(function(){
      clearInterval(timer); clearTimeout(killer);
      busy=false; stage.classList.remove("scanning"); go.textContent=t("checkBtn"); refresh();
      if(window.matchMedia("(max-width:980px)").matches){ sheet.scrollIntoView({block:"start"}); }
    });
  });

  langSel.addEventListener("change",function(){
    LANG=langSel.value; document.documentElement.lang=LANG; lsSet("bg-lang",LANG);
    go.textContent=busy?t("checking"):t("checkBtn");
    if(last && !busy && adFile){
      var note=h("p",{class:"fine",text:t("rerun")});
      if(last.kind==="fail") renderFail(last.d,note); else renderPass(last.d,note);
    }
    if(!$("p-home").hidden) renderPanel();
  });

  /* ---------- settings ---------- */
  function syncDemoSw(){ $("demoSw").setAttribute("aria-checked",DEMO_ON?"true":"false"); }
  $("demoSw").addEventListener("click",function(){ DEMO_ON=!DEMO_ON; lsSet("bg-demo",DEMO_ON?"1":"0"); syncDemoSw(); route(); toast(DEMO_ON?"Sample data is on.":"Sample data is off. You now see only your own checks."); });
  $("clearHist").addEventListener("click",function(){ if(!window.confirm("Remove all verifications saved in this browser?")) return; real=[]; lsSet("bg-history","[]"); updateDot(); toast("History cleared."); });
  $("clearRules").addEventListener("click",function(){ if(!window.confirm("Remove the company rules saved in this browser?")) return; $("policyText").value=""; showPolicyFile(null); lsDel("bg-rules-text"); lsDel("bg-rules-file"); refresh(); toast("Saved rules removed."); });
  syncDemoSw();

  /* ---------- team: cipher wheel ---------- */
  var NS="http://www.w3.org/2000/svg", shift=3;
  function ring(g,r,size,fill){
    for(var i=0;i<26;i++){
      var tx=document.createElementNS(NS,"text");
      tx.setAttribute("x",160); tx.setAttribute("y",160-r); tx.setAttribute("text-anchor","middle");
      tx.setAttribute("dominant-baseline","central"); tx.setAttribute("font-size",size); tx.setAttribute("fill",fill);
      tx.setAttribute("transform","rotate("+(i*360/26)+" 160 160)"); tx.textContent=String.fromCharCode(65+i); g.appendChild(tx);
    }
  }
  ring($("outer"),132,16,"#fff"); ring($("innerLetters"),84,15,"#fff");
  function encode(s,k){ return s.replace(/[A-Z]/g,function(c){return String.fromCharCode((c.charCodeAt(0)-65+k)%26+65);}); }
  function spin(){ $("inner").style.transform="rotate("+(shift*360/26)+"deg)"; $("enc").textContent=encode("THE CIPHERERS",shift); }
  spin();
  $("wheelbtn").addEventListener("click",function(){ shift=(shift+1)%26; if(shift===0) shift=26; spin(); });
  $("copyMail").addEventListener("click",function(){ copy("thecipherers@gmail.com",$("copyMail"),"Copied"); });

  /* ---------- confetti on a pass ---------- */
  var reduce=window.matchMedia("(prefers-reduced-motion:reduce)").matches, fx=$("fx");
  function rnd(a,b){return a+Math.random()*(b-a);}
  window.__confetti=function(){
    if(reduce) return;
    var c=fx.getContext("2d"); fx.width=window.innerWidth; fx.height=window.innerHeight;
    var sh=sheet.getBoundingClientRect();
    var ox=Math.max(40,Math.min(sh.left+sh.width/2,window.innerWidth-40));
    var oy=Math.max(90,Math.min(sh.top+130,window.innerHeight*.6));
    var colors=["#6D5DF6","#FFC247","#2ED8A3","#FF6B8A","#5EE0B5","#8E80FF"], bits=[];
    for(var n=0;n<120;n++){
      var a=-Math.PI/2+rnd(-1.15,1.15), v=rnd(6,14);
      bits.push({x:ox,y:oy,vx:Math.cos(a)*v,vy:Math.sin(a)*v,w:rnd(6,12),h:rnd(4,8),r:rnd(0,6.28),vr:rnd(-.3,.3),c:colors[n%colors.length]});
    }
    var start=null;
    function tick(ts){
      if(start===null) start=ts;
      var el=ts-start; c.clearRect(0,0,fx.width,fx.height);
      if(el>2600) return;
      bits.forEach(function(b){
        b.vy+=.3; b.vx*=.992; b.x+=b.vx; b.y+=b.vy; b.r+=b.vr;
        c.save(); c.globalAlpha=Math.max(0,1-el/2600); c.translate(b.x,b.y); c.rotate(b.r); c.fillStyle=b.c; c.fillRect(-b.w/2,-b.h/2,b.w,b.h); c.restore();
      });
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  };

  /* ---------- animated particle network ---------- */
  (function(){
    var cv=$("net"), ctx=cv.getContext("2d"), W=0,H=0,dpr=Math.min(window.devicePixelRatio||1,2), pts=[], mouse={x:-9999,y:-9999}, run=true;
    function size(){
      W=window.innerWidth; H=window.innerHeight; cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
      var n=Math.round(Math.min(70,W*H/24000)); pts=[];
      for(var k=0;k<n;k++) pts.push({x:Math.random()*W,y:Math.random()*H,vx:rnd(-.3,.3),vy:rnd(-.3,.3),r:rnd(1.2,2.6)});
    }
    function frame(){
      ctx.clearRect(0,0,W,H);
      var dk=document.documentElement.getAttribute("data-theme")==="dark", rgb=dk?"150,138,255":"109,93,246", j,k,p,q,dx,dy,dist;
      for(j=0;j<pts.length;j++){
        p=pts[j];
        if(!reduce){
          dx=mouse.x-p.x; dy=mouse.y-p.y; dist=Math.sqrt(dx*dx+dy*dy);
          if(dist<170&&dist>1){ p.vx+=dx/dist*.01; p.vy+=dy/dist*.01; }
          p.vx*=.996; p.vy*=.996;
          var sp=Math.sqrt(p.vx*p.vx+p.vy*p.vy); if(sp>.8){p.vx*=.8/sp;p.vy*=.8/sp;} if(sp<.1){p.vx+=rnd(-.02,.02);p.vy+=rnd(-.02,.02);}
          p.x+=p.vx; p.y+=p.vy;
          if(p.x<-10)p.x=W+10; if(p.x>W+10)p.x=-10; if(p.y<-10)p.y=H+10; if(p.y>H+10)p.y=-10;
        }
      }
      for(j=0;j<pts.length;j++){
        p=pts[j];
        for(k=j+1;k<pts.length;k++){
          q=pts[k]; dx=p.x-q.x; dy=p.y-q.y; dist=Math.sqrt(dx*dx+dy*dy);
          if(dist<130){ ctx.strokeStyle="rgba("+rgb+","+(0.22*(1-dist/130))+")"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke(); }
        }
        dx=p.x-mouse.x; dy=p.y-mouse.y; dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<180){ ctx.strokeStyle="rgba(255,180,84,"+(0.55*(1-dist/180))+")"; ctx.lineWidth=1.3; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(mouse.x,mouse.y); ctx.stroke(); }
        ctx.fillStyle="rgba("+rgb+",.6)"; ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,6.2832); ctx.fill();
      }
      if(!reduce&&run) requestAnimationFrame(frame);
    }
    size(); frame();
    var rt; window.addEventListener("resize",function(){ clearTimeout(rt); rt=setTimeout(function(){ size(); if(reduce) frame(); },150); });
    window.addEventListener("pointermove",function(e){ mouse.x=e.clientX; mouse.y=e.clientY; });
    document.addEventListener("visibilitychange",function(){ run=!document.hidden; if(run&&!reduce) requestAnimationFrame(frame); });
  })();

  /* ---------- start ---------- */
  go.textContent=t("checkBtn");
  emptyState(); updateDot(); route();
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
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#F4F5FB;--surface:#fff;--ink:#111A36;--muted:#5A6482;--line:#DCE0EE;--brand:#4A3FE0;--brand-soft:#E9E7FD;
  --pass:#12805F;--pass-soft:#DFF4EC;--fail:#C7372B;--focus:#1F6FEB;
  --display:"Plus Jakarta Sans","Instrument Sans",system-ui,sans-serif;--body:"Instrument Sans",system-ui,sans-serif;
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

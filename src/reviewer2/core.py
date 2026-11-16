import os
import time
import shutil
import tempfile
import subprocess
import re
import atexit
import threading
from datetime import datetime
from pathlib import Path
from pypdf import PdfWriter, PdfReader

from reviewer2.paths import prompt_path

# --- SDK IMPORT ---
import anthropic

# Global Usage Log
USAGE_LOG = []

# Set per-run by the pipeline so each usage row can be attributed to a run.
CURRENT_JOB_UUID = None

# GLOBAL CACHE REGISTRY
# Format: { "path/to/local/file.pdf": file_id }
_FILE_CACHE = {}
# Stages now run concurrently, so the first upload of a shared PDF must be
# serialized per path or every parallel stage re-uploads the same 50MB file.
_CACHE_LOCK = threading.Lock()
_UPLOAD_LOCKS = {}

# Stage code names a tier per call. Every tier runs on Claude Opus 5.5; the
# tiers differ in reasoning effort instead (see _effort_for). To run a tier on
# another model, change it here or set REVIEWER2_MODEL_OVERRIDE to a tier key
# or a Claude model ID.
DEFAULT_MODEL = "claude-opus-5-5"
MODELS = {
    "flash_lite": DEFAULT_MODEL,
    "flash_lite_3": DEFAULT_MODEL,
    "flash_2_5": DEFAULT_MODEL,
    "flash_3": DEFAULT_MODEL,
    "pro_2_5": DEFAULT_MODEL,
    "pro_3": DEFAULT_MODEL,
    "pro_3_1": DEFAULT_MODEL,
}

# Models that accept the server-side refusal fallback. On a safety-classifier
# decline the API re-runs the same request on a fallback model in one call.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# The PDF prefix is reused by dozens of stages over a run that can last 45
# minutes, longer than the default 5-minute cache lifetime.
_PDF_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}

# Default output ceiling. Thinking counts toward it, so callers' small caps
# (written for answer length alone) are raised to at least _MIN_MAX_TOKENS.
_DEFAULT_MAX_TOKENS = 64000
_MIN_MAX_TOKENS = 16000
_MAX_CONTINUATIONS = 5


def _model_override():
    return os.environ.get("REVIEWER2_MODEL_OVERRIDE")


def _effort_for(thinking_level, thinking_budget):
    """Translate the Gemini-era thinking knobs to a Claude effort level.

    thinking_level ("low" / "medium" / "high") maps one-to-one. A fixed
    thinking_budget (0, 512, 1000: short clerical stages) maps to "low"; the
    dynamic budget (-1) and no setting map to "medium".
    """
    if thinking_level in ("low", "medium", "high", "xhigh", "max"):
        return thinking_level
    if thinking_budget is not None and int(thinking_budget) >= 0:
        return "low"
    return "medium"


def _get_client(timeout):
    # Finite read timeout: a silently dead connection (e.g. local link drop)
    # must surface as an exception for the retry loop, not block forever.
    # Retries are handled by call_claude's own loop, so the SDK's are off.
    return anthropic.Anthropic(timeout=timeout, max_retries=0)


def get_or_upload_file(client, file_path, force_upload=False):
    """
    Checks if file is already uploaded in this session.
    If yes, returns existing file ID. If no (or force=True), uploads and caches it.
    """
    global _FILE_CACHE

    with _CACHE_LOCK:
        if not force_upload and file_path in _FILE_CACHE:
            return _FILE_CACHE[file_path]
        path_lock = _UPLOAD_LOCKS.setdefault(file_path, threading.Lock())

    with path_lock:
        # Another thread may have finished the upload while we waited.
        if not force_upload and file_path in _FILE_CACHE:
            return _FILE_CACHE[file_path]
        return _upload_file(client, file_path)


def _upload_file(client, file_path):
    # Shrink oversized PDFs by stripping photographic figures (which the
    # finders ignore) while preserving text and tables. No-op for PDFs already
    # under the target. Lazy import so a missing PyMuPDF degrades to uploading
    # the original rather than breaking core.py at import time.
    upload_path = file_path
    if isinstance(file_path, str) and file_path.lower().endswith(".pdf"):
        try:
            from reviewer2.pdf_compress import compress_pdf_for_gemini
            compressed, cinfo = compress_pdf_for_gemini(file_path)
            if cinfo.get("action") == "stripped":
                print(f"    ↳ 🗜  PDF {cinfo['orig_bytes']//1048576}MB → "
                      f"{cinfo['final_bytes']//1048576}MB "
                      f"(stripped {cinfo['stripped']} figure image(s); "
                      f"{cinfo['protected']} protected)")
                if cinfo.get("over_limit"):
                    print("    ⚠ still over the 50MB target after stripping; "
                          "upload may be rejected")
            upload_path = compressed
        except Exception as e:
            print(f"    ⚠ PDF compression skipped ({e}); uploading original")
            upload_path = file_path

    print(f"    ↳ 📤 Uploading NEW file to Claude Files API: {os.path.basename(file_path)}...")
    try:
        uploaded = client.files.upload(
            file=(os.path.basename(file_path), Path(upload_path).read_bytes(), "application/pdf"))
        _FILE_CACHE[file_path] = uploaded.id
        print(f"    ✓ Upload complete: {uploaded.id} (Cached)")
        return uploaded.id
    except Exception as e:
        # If upload fails, ensure we don't leave bad cache
        if file_path in _FILE_CACHE:
            del _FILE_CACHE[file_path]
        raise IOError(f"Error uploading PDF '{file_path}': {e}")


def cleanup_resources():
    """
    Deletes all files uploaded during this session.
    """
    global _FILE_CACHE
    if not _FILE_CACHE:
        return

    print("\n🧹 Cleaning up Claude Files API uploads...")
    try:
        client = _get_client(60)
        for path, file_id in list(_FILE_CACHE.items()):
            try:
                client.files.delete(file_id)
                print(f"  ✓ Deleted remote file: {file_id}")
            except anthropic.NotFoundError:
                pass  # already gone
            except Exception as e:
                print(f"  ⚠ Failed to delete {file_id}: {e}")
        _FILE_CACHE.clear()
    except Exception as e:
        print(f"  ⚠ Cleanup initialization failed: {e}")

# Register cleanup to run automatically on exit
atexit.register(cleanup_resources)


def _log_usage(response, step):
    usage = getattr(response, "usage", None)
    if not usage:
        return
    USAGE_LOG.append({
        "model_name": getattr(response, "model", None) or DEFAULT_MODEL,
        "input_tokens": usage.input_tokens or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        # Claude's output_tokens already includes thinking.
        "output_tokens": usage.output_tokens or 0,
        "thoughts_tokens": 0,
        "timestamp": datetime.now().isoformat(),
        "step": step or "unknown",
        "job_uuid": CURRENT_JOB_UUID,
    })


class _Fatal(RuntimeError):
    """An error that retrying cannot fix."""


def call_claude(prompt=None, pdf_file_path=None, model_type="flash_lite",
                temperature=None, thinking_level=None, thinking_budget=None,
                media_resolution=None, system_instruction=None,
                max_retries=10, retry_forever_on_rate_limit=True,
                step=None, use_search=False, max_output_tokens=None,
                use_code_exec=False, timeout=1800):
    """
    Call Claude with retry logic and return the response text.

    The PDF is uploaded once per run through the Files API and cached in the
    prompt cache, so the stages that share it pay for it once per hour.

    ``temperature`` and ``media_resolution`` are accepted for compatibility
    with the stage code and ignored: current Claude models reject sampling
    parameters, and PDF pages are read at a fixed resolution.
    """
    model_type = _model_override() or model_type

    # Env-gated code execution: comma-separated step names in CODE_EXEC_STEPS
    if step and step in os.environ.get("CODE_EXEC_STEPS", "").split(","):
        use_code_exec = True

    model_name = MODELS.get(model_type, model_type)
    if not model_name.startswith("claude-"):
        print(f"  ⚠  Warning: Unknown model_type '{model_type}', falling back to {DEFAULT_MODEL}")
        model_name = DEFAULT_MODEL

    client = _get_client(timeout)

    tools = None
    if use_search:
        tools = [{"type": "web_search_20260209", "name": "web_search"}]
    elif use_code_exec:
        # Python sandbox for numerical checks (the 01fe defence)
        tools = [{"type": "code_execution_20260521", "name": "code_execution"}]

    params = {
        "model": model_name,
        "max_tokens": max(max_output_tokens or _DEFAULT_MAX_TOKENS, _MIN_MAX_TOKENS),
        "output_config": {"effort": _effort_for(thinking_level, thinking_budget)},
    }
    if system_instruction:
        params["system"] = system_instruction
    if tools:
        params["tools"] = tools
    if model_name in _FALLBACK_MODELS:
        params["betas"] = [_FALLBACK_BETA]
        params["fallbacks"] = "default"

    attempt = 0
    base_delay = 5
    max_delay = 300
    force_reupload = False

    while True:
        attempt += 1
        try:
            # Build the user turn inside the loop so a re-upload takes effect.
            content = []
            if pdf_file_path:
                file_id = get_or_upload_file(client, pdf_file_path, force_upload=force_reupload)
                force_reupload = False
                content.append({
                    "type": "document",
                    "source": {"type": "file", "file_id": file_id},
                    "cache_control": _PDF_CACHE_CONTROL,
                })
            if prompt:
                content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            # Server tools can pause a long turn; resend to let it continue.
            for _ in range(_MAX_CONTINUATIONS + 1):
                with client.beta.messages.stream(messages=messages, **params) as stream:
                    response = stream.get_final_message()
                _log_usage(response, step)
                if response.stop_reason != "pause_turn":
                    break
                messages = messages[:1] + [{"role": "assistant", "content": response.content}]

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                raise _Fatal(f"FATAL: Claude declined the request ({getattr(details, 'category', None)}).")

            result = "".join(b.text for b in response.content if b.type == "text").strip()
            if not result:
                if response.stop_reason == "max_tokens":
                    raise _Fatal("FATAL: Model hit max_tokens and returned no text.")
                raise ValueError(f"API returned empty text (stop_reason={response.stop_reason})")
            return result

        except _Fatal:
            raise

        except IOError as e:
            print(f"    ⚠  Upload failed inside retry loop: {e}")
            if attempt >= max_retries:
                raise RuntimeError(f"PDF upload failed after {attempt} attempts: {e}")
            time.sleep(5)
            continue

        except anthropic.NotFoundError as e:
            # A cached file can expire or be deleted mid-run: re-upload once.
            if pdf_file_path and attempt == 1:
                print("    ⚠  File not found (404). Re-uploading on next attempt.")
                with _CACHE_LOCK:
                    _FILE_CACHE.pop(pdf_file_path, None)
                force_reupload = True
                continue
            raise RuntimeError(f"FATAL: Not found: {e}")

        except (anthropic.BadRequestError, anthropic.AuthenticationError,
                anthropic.PermissionDeniedError) as e:
            raise RuntimeError(f"FATAL: Claude API rejected the request: {e}")

        except anthropic.RateLimitError as e:
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            retry_after = e.response.headers.get("retry-after") if e.response is not None else None
            if retry_after and retry_after.isdigit():
                delay = max(delay, int(retry_after))
            print(f"    ↳ Rate Limited (429) on attempt {attempt}. Waiting {delay}s...")
            if not retry_forever_on_rate_limit and attempt >= max_retries:
                raise RuntimeError(f"Rate limit exceeded after {attempt} attempts: {e}")
            time.sleep(delay)
            continue

        except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            # Connection drops, timeouts, 5xx and 529 overloaded.
            status = getattr(e, "status_code", None)
            if status is not None and status < 500:
                raise RuntimeError(f"FATAL: Claude API error {status}: {e}")
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"    ⚠  Server/Network Error (Attempt {attempt}): {str(e)[:200]}. Waiting {delay}s...")
            if not retry_forever_on_rate_limit and attempt >= max_retries:
                raise RuntimeError(f"Server error after {attempt} attempts: {e}")
            time.sleep(delay)
            continue

        except Exception as e:
            delay = min(base_delay * (2 ** (attempt - 1)), 30)
            print(f"    ⚠  API Error (Attempt {attempt}/{max_retries}): {str(e)[:200]}")
            if attempt >= max_retries:
                raise RuntimeError(f"Claude API call failed after {attempt} attempts. Last error: {e}")
            time.sleep(delay)


# =============================================================================
# OPTIONAL PROVIDER ROUTING
# =============================================================================
# When DEEPSEEK_API_KEY is set, these clerical stages go to DeepSeek V4 Flash,
# which is much cheaper for them. They work purely on prior text: DeepSeek has
# no PDF vision, so ONLY stages whose prompt is fully self-contained in text
# belong here. 00b_metadata_clean's prompt says "rely ONLY on the INPUT DATA",
# so it qualifies. Without the key, everything runs on Claude.
DS_FLASH_STAGES = {
    "00b_metadata_clean",
    "00b_2_metadata_math",
    "02b_compiler_1",
    "02d_compiler_2",
    "02f_compiler_3",
    "06_legal",
}


def call_model(prompt=None, pdf_file_path=None, model_type="flash_lite",
               step=None, **kwargs):
    """Dispatcher used by the bureaucratic stages. Routes DS_FLASH_STAGES to
    DeepSeek V4 Flash, everything else to Claude. A REVIEWER2_MODEL_OVERRIDE
    keeps everything on Claude. The DeepSeek path drops the PDF — these stages
    are text-only by construction — and falls back to Claude on any DeepSeek
    error so a bureaucratic stage can never sink a report."""
    route_ds = (step in DS_FLASH_STAGES and not _model_override()
                and os.environ.get("DEEPSEEK_API_KEY"))
    if route_ds:
        try:
            from reviewer2.deepseek_client import call_deepseek
            return call_deepseek(
                prompt,
                model="deepseek-v4-flash",
                temperature=kwargs.get("temperature", 0.0),
                thinking_budget=kwargs.get("thinking_budget"),
                thinking_level=kwargs.get("thinking_level"),
                system_instruction=kwargs.get("system_instruction"),
                step=step,
                max_output_tokens=kwargs.get("max_output_tokens"),
            )
        except Exception as e:
            print(f"    ⚠  DeepSeek failed for {step} ({e}); falling back to Claude")
    return call_claude(prompt, pdf_file_path, model_type=model_type, step=step, **kwargs)


def save_output(content, filename, output_dir):
    if content is None:
        print(f"  ⚠  WARNING: content is None for {filename}. Creating empty file.")
        content = ""
        
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Saved: {filename}")


def load_prompt(relative_path):
    path = prompt_path(relative_path)
    if not os.path.exists(path):
        return f"[Error: Prompt file {relative_path} missing]"

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if "{{OUTPUT_FORMAT}}" in text:
        output_format_path = prompt_path("resources/output_format.txt")
        if os.path.exists(output_format_path):
            with open(output_format_path, "r", encoding="utf-8") as f:
                fmt = f.read()
            text = text.replace("{{OUTPUT_FORMAT}}", fmt)
        else:
            text = text.replace("{{OUTPUT_FORMAT}}", "")
    
    return text


def sanitize_pdf_ghostscript(input_path):
    if os.path.getsize(input_path) == 0:
        return input_path

    fd, fixed_path = tempfile.mkstemp(suffix=".pdf", prefix="sanitized_")
    os.close(fd)
    
    gs_cmd = shutil.which("gs")
    if gs_cmd:
        cmd = [
            gs_cmd, "-o", fixed_path, 
            "-sDEVICE=pdfwrite", 
            "-dPDFSETTINGS=/prepress", 
            "-dQUIET", 
            input_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
                return fixed_path
        except subprocess.CalledProcessError:
            pass
    
    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.write(fixed_path)
        return fixed_path
    except Exception:
        pass
        
    return input_path


def merge_pdfs_python(main_pdf, supplement_source, output_dir=None):
    # 1. Determine Output Path
    # We want to overwrite the original or keep the same filename in the target dir.
    # No more "_merged" suffix.
    if output_dir:
        output_path = os.path.join(output_dir, os.path.basename(main_pdf))
    else:
        output_path = main_pdf

    # Write to a temp file first to avoid read/write conflicts if overwriting
    temp_path = output_path + ".tmp_merging"

    writer = PdfWriter()
    page_info = {'pages': []}
    
    supplement_pdfs = []
    if isinstance(supplement_source, list):
        supplement_pdfs = supplement_source
    elif isinstance(supplement_source, str) and os.path.isdir(supplement_source):
        files = sorted(os.listdir(supplement_source))
        supplement_pdfs = [os.path.join(supplement_source, f) for f in files if f.lower().endswith(".pdf")]
    
    try:
        reader = PdfReader(main_pdf)
        main_page_count = len(reader.pages)
        for page in reader.pages:
            writer.add_page(page)
        
        page_info['pages'].append({
            'name': 'Main document',
            'pages': main_page_count,
            'numbering': 'See main text page numbers',
            'pdf_start': 1,
            'pdf_end': main_page_count
        })
    except Exception as e:
        print(f"  ✗ Error merging main PDF: {e}")
        # If main PDF fails, we can't proceed safely
        return main_pdf, page_info

    current_pdf_page = main_page_count + 1
    
    # --- INSERT SEPARATOR IF SUPPLEMENTS EXIST ---
    if supplement_pdfs:
        sep_path = str(prompt_path("resources/separator_supp.pdf"))
        if os.path.exists(sep_path):
            try:
                sep_reader = PdfReader(sep_path)
                sep_count = len(sep_reader.pages)
                for page in sep_reader.pages:
                    writer.add_page(page)
                
                # IMPORTANT: Increment the PDF page counter so the next section
                # (the actual supplements) has the correct start page in the index.
                current_pdf_page += sep_count
            except Exception as e:
                print(f"  ⚠ Warning: Could not merge separator: {e}")
        else:
             print(f"  ⚠ Warning: Separator file not found at {sep_path}")

    # --- APPEND SUPPLEMENTS ---
    for i, supp_pdf in enumerate(supplement_pdfs, 1):
        try:
            reader = PdfReader(supp_pdf)
            supp_page_count = len(reader.pages)
            
            for page in reader.pages:
                writer.add_page(page)
            
            first_page_text = ""
            try: first_page_text = reader.pages[0].extract_text()
            except: pass
            
            numbering = "Unknown"
            if re.search(r'\bS\d+\b', first_page_text):
                numbering = "S-numbered"
            elif re.search(r'\bAppendix\s+[A-Z]\b', first_page_text, re.IGNORECASE):
                numbering = "Appendix-lettered"
            
            filename = os.path.basename(supp_pdf)
            # Strip the staging prefixes (10_10_, then an NNN__ ordering
            # prefix if present), leaving the original name.
            display_name = re.sub(r'^\d+_\d+_', '', filename)
            display_name = re.sub(r'^\d+__', '', display_name)
            
            page_info['pages'].append({
                'name': display_name,
                'pages': supp_page_count,
                'numbering': numbering,
                'pdf_start': current_pdf_page,
                'pdf_end': current_pdf_page + supp_page_count - 1
            })
            current_pdf_page += supp_page_count
            
        except Exception as e:
            print(f"  ⚠  Skipped corrupt supplement {supp_pdf}: {e}")

    # --- SAVE AND MOVE ---
    with open(temp_path, "wb") as f:
        writer.write(f)
    
    # Overwrite the target (or source) with the new merged version
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(temp_path, output_path)
    
    return output_path, page_info


def format_page_mapping_for_prompt(page_mapping):
    # New deterministic map (pagemap.build_page_map): render the real
    # physical->printed sections + appendix ceiling.
    if isinstance(page_mapping, dict) and ('sections' in page_mapping or 'confidence' in page_mapping):
        from reviewer2 import pagemap
        return pagemap.render_details(page_mapping)
    # Legacy shape (kept for safety; not produced anywhere now).
    if not page_mapping.get('ranges'):
        return "- Document structure: Check printed page numbers in the document"
    lines = []
    for range_info in page_mapping['ranges']:
        line = f"- {range_info['section']}: PDF viewer pages {range_info['pdf_pages']} → Printed pages: {range_info['printed_pages']}"
        lines.append(line)
    return "\n".join(lines)


def load_page_numbers_snippet(page_mapping):
    template_path = str(prompt_path("resources/page_numbers.txt"))
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        print(f"  ⚠  Warning: {template_path} not found. Using basic instructions.")
        return "\n\n## PAGE NUMBERS\n- Use printed page numbers from the document, not PDF viewer page numbers"
    
    structure_details = format_page_mapping_for_prompt(page_mapping)
    formatted = template.replace("{{PAGE_STRUCTURE_DETAILS}}", structure_details)
    
    return "\n\n" + formatted
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bibtexparser==1.4.1",
#     "click==8.3.1",
#     "python-dotenv==1.2.1",
#     "pydantic>=2",
#     "requests==2.32.3",
#     "rich==14.3.2",
#     "marker-pdf==1.10.2",  # pinned: we patch OllamaService internals (see load_marker)
# ]
# ///

"""
corpus_to_md.py
---------------
Parses a BibTeX file, resolves a PDF for each entry, and converts
them into high-fidelity Markdown using datalab-to/marker.

PDF resolution is source-agnostic and tries, in order:
  1. A local drop-in at ``{pdf-dir}/{citekey}.pdf``.
  2. A ``file = {...}`` field in the bib entry (Zotero, JabRef, ...).
  3. A bib ``url`` field that points directly at a ``.pdf``.
  4. An arXiv id (``eprint`` field, or inferred from url/doi).
  5. Unpaywall Open-Access resolution via the entry ``doi``.
  6. An interactive prompt asking the user for a local path.

The bib file therefore serves two roles: citation metadata and a
manifest of where each PDF lives. Nothing here is tied to Zotero.

Features:
- Configurable torch device selection.
- OCR and LLM toggles for layout/math accuracy.
- Multi-provider vision LLM (Gemini/OpenAI/Anthropic/Ollama/Azure) via .env.
- Vision-capability check before the expensive marker pass.
- Mathematical precision checking (chktex integration).
- Internal link stripping (preserves external).
- Optional figure captioning, reusing the chosen provider.
"""

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import bibtexparser
import click
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()

# marker is imported lazily (see load_marker) because marker and its surya
# backend read TORCH_DEVICE at import time. Importing here would lock the device
# before --device is parsed, so --device cpu would be silently ignored.
ConfigParser: Any = None
PdfConverter: Any = None
create_model_dict: Any = None


# marker's BaseService.img_to_base64 defaults to WEBP. Ollama's llama.cpp
# backend (qwen3-vl, llama3.2-vision, ...) decodes images with stb_image, which
# has no WEBP support, so those payloads fail server-side. PNG is decoded
# natively by stb_image and is lossless, preserving transparency in marker's
# extracted figures. Override via env (e.g. JPEG) for smaller payloads.
OLLAMA_IMAGE_FORMAT = os.getenv("OLLAMA_IMAGE_FORMAT", "PNG").upper()


def _ollama_process_images(self: Any, images: list) -> list:
    """Re-encodes figures in an Ollama-decodable format instead of WEBP."""
    return [self.img_to_base64(img, format=OLLAMA_IMAGE_FORMAT) for img in images]


def _parse_ollama_json(raw: str) -> dict:
    """Extracts the JSON object from an Ollama generation body.

    Tolerates a ``<think>...</think>`` wrapper and ```` ```json ```` fences,
    and falls back to the first balanced ``{...}`` span when the model prepends
    stray reasoning text.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _ollama_call(
    self: Any,
    prompt: str,
    image: Any,
    block: Any,
    response_schema: Any,
    max_retries: int | None = None,
    timeout: int | None = None,
) -> dict:
    """Drop-in for OllamaService.__call__ that fixes two llama.cpp failures.

    1. Forwards the *complete* JSON schema (including ``$defs``). marker's stock
       service copies only ``properties``/``required``, stripping ``$defs`` and
       leaving a dangling ``$ref`` for nested schemas (section-header, page
       correction); llama.cpp's grammar converter then rejects it with HTTP 400.
    2. Reads the structured output from ``thinking`` when ``response`` is empty.
       qwen3-vl emits its JSON into the reasoning channel, so marker's stock
       ``json.loads(response_data["response"])`` raises "Expecting value".
    """
    url = f"{self.ollama_base_url}/api/generate"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": self.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": response_schema.model_json_schema(),
        "images": self.format_image_for_llm(image),
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()

        if block:
            used = response_data.get("prompt_eval_count", 0) + response_data.get(
                "eval_count", 0
            )
            block.update_metadata(llm_request_count=1, llm_tokens_used=used)

        data = (response_data.get("response") or "").strip() or (
            response_data.get("thinking") or ""
        ).strip()
        if not data:
            console.log("[Warning] Ollama returned an empty response.")
            return {}
        return _parse_ollama_json(data)
    except Exception as e:
        console.log(f"[Warning] Ollama inference failed: {e}")

    return {}


def load_marker() -> None:
    """Imports marker into module globals. Call only after TORCH_DEVICE is set."""
    global ConfigParser, PdfConverter, create_model_dict
    try:
        from marker.config.parser import ConfigParser as _CP
        from marker.converters.pdf import PdfConverter as _PC
        from marker.models import create_model_dict as _CMD
        from marker.services.ollama import OllamaService
    except ImportError:
        console.print(
            "Error: datalab-to/marker not found. Install with: pip install marker-pdf"
        )
        sys.exit(1)
    ConfigParser, PdfConverter, create_model_dict = _CP, _PC, _CMD

    # Patch the class (not an instance): marker resolves the service by dotted
    # path for both our caption_service and its internal use_llm pass, so this
    # covers every code path that sends figures to Ollama.
    OllamaService.process_images = _ollama_process_images
    OllamaService.__call__ = _ollama_call


# ---------------------------------------------------------------------------
# LLM provider selection & figure captioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    """Maps a friendly provider name onto marker's service config.

    ``service`` is the dotted path to marker's BaseService subclass.
    ``key_env``/``key_config`` and ``base_url_env``/``base_url_config`` wire
    an environment variable onto the flat config key the service reads.
    ``vision_patterns`` are regexes of model families known to be multimodal,
    used as a heuristic check for cloud providers (Ollama is checked live).
    """

    service: str
    model_key: str
    default_model: str
    key_env: str | None = None
    key_config: str | None = None
    base_url_env: str | None = None
    base_url_config: str | None = None
    extra_env: tuple[tuple[str, str], ...] = ()
    vision_patterns: tuple[str, ...] = ()


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        service="marker.services.gemini.GoogleGeminiService",
        model_key="gemini_model_name",
        default_model="gemini-2.0-flash",
        key_env="GOOGLE_API_KEY",
        key_config="gemini_api_key",
        vision_patterns=("gemini-",),
    ),
    "openai": Provider(
        service="marker.services.openai.OpenAIService",
        model_key="openai_model",
        default_model="gpt-4o-mini",
        key_env="OPENAI_API_KEY",
        key_config="openai_api_key",
        base_url_env="OPENAI_BASE_URL",
        base_url_config="openai_base_url",
        vision_patterns=(
            "gpt-4o",
            "gpt-4\\.1",
            "gpt-4-turbo",
            "gpt-4-vision",
            "o1",
            "o3",
            "o4",
        ),
    ),
    "anthropic": Provider(
        service="marker.services.claude.ClaudeService",
        model_key="claude_model_name",
        default_model="claude-sonnet-4-6",
        key_env="ANTHROPIC_API_KEY",
        key_config="claude_api_key",
        vision_patterns=("claude-3", "claude-(opus|sonnet|haiku)-4"),
    ),
    "ollama": Provider(
        service="marker.services.ollama.OllamaService",
        model_key="ollama_model",
        default_model="llama3.2-vision:11b",
        base_url_env="OLLAMA_BASE_URL",
        base_url_config="ollama_base_url",
    ),
    "azure": Provider(
        service="marker.services.azure_openai.AzureOpenAIService",
        model_key="deployment_name",
        default_model="",
        key_env="AZURE_API_KEY",
        key_config="azure_api_key",
        extra_env=(
            ("AZURE_ENDPOINT", "azure_endpoint"),
            ("AZURE_API_VERSION", "azure_api_version"),
        ),
    ),
}


def build_llm_config(
    name: str, model_override: str | None
) -> tuple[dict[str, Any], str, str | None]:
    """Builds marker config keys for a provider; returns (config, model, base_url).

    Raises ``click.ClickException`` if a required key or model is missing.
    """
    provider = PROVIDERS[name]
    model = model_override or provider.default_model
    if not model:
        raise click.ClickException(f"--llm-model is required for provider '{name}'.")

    config: dict[str, Any] = {
        "llm_service": provider.service,
        provider.model_key: model,
    }

    if provider.key_config:
        key = os.getenv(provider.key_env) if provider.key_env else None
        if not key:
            raise click.ClickException(
                f"Provider '{name}' needs an API key. "
                f"Set {provider.key_env} in your environment or .env."
            )
        config[provider.key_config] = key

    base_url = None
    if provider.base_url_config and provider.base_url_env:
        base_url = os.getenv(provider.base_url_env)
        if base_url:
            config[provider.base_url_config] = base_url

    for env_var, cfg_key in provider.extra_env:
        val = os.getenv(env_var)
        if not val:
            raise click.ClickException(f"Provider '{name}' needs {env_var} set.")
        config[cfg_key] = val

    return config, model, base_url


def build_service(name: str, config: dict[str, Any]) -> Any:
    """Instantiates marker's BaseService for ``name`` from a marker config dict."""
    module_path, cls_name = PROVIDERS[name].service.rsplit(".", 1)
    service_cls = getattr(importlib.import_module(module_path), cls_name)
    return service_cls(config)


def is_vision_model(
    name: str, model: str, base_url: str | None
) -> tuple[bool | None, str]:
    """Checks whether ``model`` is vision-capable.

    Returns ``(ok, detail)`` where ok is True/False, or None when it cannot be
    verified. Ollama is authoritative (``/api/show`` capabilities); cloud
    providers fall back to a known-vision-family heuristic.
    """
    if name == "ollama":
        url = (base_url or "http://localhost:11434").rstrip("/") + "/api/show"
        try:
            resp = requests.post(url, json={"model": model}, timeout=10)
            resp.raise_for_status()
            caps = resp.json().get("capabilities", []) or []
        except Exception as e:
            return None, f"could not query Ollama at {url}: {e}"
        if "vision" in caps:
            return True, f"capabilities={caps}"
        return False, f"capabilities={caps} (no 'vision')"

    patterns = PROVIDERS[name].vision_patterns
    if any(re.search(p, model, re.I) for p in patterns):
        return True, "matches a known vision-capable family"
    return None, "not in the known vision-model list"


class CaptionSchema(BaseModel):
    caption: str


def caption_image(service: Any, image: Any) -> str:
    """Captions a single PIL figure by reusing the chosen marker service."""
    prompt = (
        "You are a scientific document analyst. Describe this figure for a "
        "radio-astronomy research corpus: state what it shows, the axes/units, "
        "and the key takeaway. Return only the description."
    )
    try:
        result = service(prompt, image, None, CaptionSchema)
        return (result.get("caption") or "").strip() or "No description generated."
    except Exception as e:
        return f"Captioning failed: {e}"


# ---------------------------------------------------------------------------
# Utility Functions: Formatting
# ---------------------------------------------------------------------------


def check_latex_with_chktex(math_block: str) -> str | None:
    """Passes a math block to chktex. Returns warnings if bad syntax is detected."""
    if not shutil.which("chktex"):
        return None

    try:
        process = subprocess.run(
            ["chktex", "-q", "-v0"],
            input=math_block.encode("ascii"),
            capture_output=True,
            timeout=5,
        )
        output = process.stdout.decode("ascii").strip()
        if output:
            return output
    except Exception:
        pass
    return None


def post_process_markdown(md_text: str) -> str:
    """Refines the Marker output by removing internal links and verifying LaTeX."""

    def replace_link(match: re.Match) -> str:
        text = match.group(1)
        url = match.group(2)
        if url.startswith("#"):
            return text
        return match.group(0)

    md_text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, md_text)

    math_blocks = re.findall(r"\$\$(.*?)\$\$", md_text, re.DOTALL)
    for block in math_blocks:
        if not all(ord(c) < 128 for c in block):
            console.log(
                "[Warning] Unicode character found in math block. Conversion to pure LaTeX recommended."
            )

        warnings = check_latex_with_chktex(block)
        if warnings:
            console.log(f"[Warning] LaTeX validation issue in block: {warnings}")

    return md_text


# ---------------------------------------------------------------------------
# Acquisition Logic: source-agnostic PDF resolution
# ---------------------------------------------------------------------------


@dataclass
class Resolution:
    """How one BibTeX entry resolved to a PDF (or did not)."""

    entry: dict[str, str]
    path: Path | None = None
    source: str = "unresolved"

    @property
    def citekey(self) -> str:
        return self.entry.get("ID", "unknown")

    @property
    def resolved(self) -> bool:
        return self.path is not None


class PdfResolver:
    """Resolves a local PDF path for each BibTeX entry.

    Resolution is offline-first: local paths are tried before any
    network access, and Unpaywall is only consulted when nothing
    local matches and a DOI is available.
    """

    def __init__(self, pdf_dir: Path, email: str | None):
        self.pdf_dir = pdf_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        if not self.email:
            console.log(
                "[Warning] UNPAYWALL_EMAIL not set. Unpaywall DOI resolution is disabled."
            )

    def parse_bibtex(self, bib_path: Path) -> list[dict[str, str]]:
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_database = bibtexparser.load(f)
        return bib_database.entries

    @staticmethod
    def _parse_file_field(raw: str) -> list[Path]:
        """Extracts PDF paths from a bib ``file`` field.

        Handles both a bare path and the JabRef ``name:path:mimetype``
        form, and tolerates multiple ``;``-separated entries.
        """
        paths: list[Path] = []
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            # JabRef-style "name:/the/path:application/pdf" -> middle segment.
            match = re.match(r"^[^:]*:(.+):[^:/\\]+$", part)
            path = match.group(1) if match else part
            paths.append(Path(os.path.expanduser(path)))
        return paths

    @staticmethod
    def _arxiv_id(entry: dict[str, str]) -> str | None:
        """Finds an arXiv id from an explicit eprint, the url, or the DOI."""
        if entry.get("eprint"):
            return entry["eprint"]
        url = entry.get("url", "")
        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+|[a-z\-]+/[0-9]+)", url, re.I
        )
        if match:
            return match.group(1)
        doi = entry.get("doi", "")
        match = re.search(r"10\.48550/arxiv\.(.+)$", doi, re.I)
        if match:
            return match.group(1)
        return None

    def _local_candidates(self, entry: dict[str, str]) -> Iterator[Path]:
        """Yields candidate local PDF paths, drop-in convention first."""
        citekey = entry.get("ID", "")
        if citekey:
            yield self.pdf_dir / f"{citekey}.pdf"
        yield from self._parse_file_field(entry.get("file", ""))

    def _download(self, url: str, dest_path: Path) -> bool:
        """Streams a URL to disk, accepting it as a PDF by content-type or magic bytes."""
        try:
            time.sleep(1)  # Respect academic server rate limits.
            headers = {
                "User-Agent": f"CorpusIngestionBot/1.0 (mailto:{self.email or 'anonymous@example.com'})"
            }
            with requests.get(
                url, stream=True, headers=headers, timeout=30, allow_redirects=True
            ) as r:
                r.raise_for_status()
                chunks = r.iter_content(chunk_size=8192)
                first = next(chunks, b"")

                content_type = r.headers.get("Content-Type", "").lower()
                is_pdf = "application/pdf" in content_type or first[:5] == b"%PDF-"
                if not is_pdf:
                    return False

                with open(dest_path, "wb") as f:
                    f.write(first)
                    for chunk in chunks:
                        f.write(chunk)
            return True
        except Exception:
            dest_path.unlink(missing_ok=True)  # Drop any partial download.
            return False

    def _unpaywall_pdf_url(self, doi: str) -> str | None:
        """Returns a direct OA PDF url for a DOI, or None.

        ``best_oa_location`` frequently lacks ``url_for_pdf`` even when a
        PDF exists, so every ``oa_locations`` entry is scanned and a
        repository copy (arXiv, institutional) is preferred over the
        publisher's, which is the copy most likely to be a real PDF.
        """
        try:
            resp = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": self.email},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            locations = resp.json().get("oa_locations") or []
        except Exception:
            return None

        with_pdf = [loc for loc in locations if loc.get("url_for_pdf")]
        with_pdf.sort(key=lambda loc: loc.get("host_type") != "repository")
        return with_pdf[0]["url_for_pdf"] if with_pdf else None

    def resolve(self, entry: dict[str, str]) -> Resolution:
        """Resolves an entry to a :class:`Resolution` (path is None if unresolved)."""
        # 1 + 2: local drop-in and the bib ``file`` field.
        for candidate in self._local_candidates(entry):
            if candidate.is_file() and candidate.suffix.lower() == ".pdf":
                return Resolution(entry, candidate, f"local ({candidate})")

        citekey = entry.get("ID", "unknown_entry").replace("/", "_")
        dest_path = self.pdf_dir / f"{citekey}.pdf"

        # 3: a bib url that points straight at a PDF.
        url = entry.get("url", "")
        if url.lower().endswith(".pdf") and self._download(url, dest_path):
            return Resolution(entry, dest_path, "url")

        # 4: arXiv.
        arxiv_id = self._arxiv_id(entry)
        if arxiv_id and self._download(
            f"https://arxiv.org/pdf/{arxiv_id}.pdf", dest_path
        ):
            return Resolution(entry, dest_path, f"arxiv ({arxiv_id})")

        # 5: Unpaywall by DOI.
        doi = entry.get("doi")
        if doi and self.email:
            oa_url = self._unpaywall_pdf_url(doi)
            if oa_url and self._download(oa_url, dest_path):
                return Resolution(entry, dest_path, "unpaywall")

        return Resolution(entry)


# ---------------------------------------------------------------------------
# Core Conversion Logic
# ---------------------------------------------------------------------------


class CorpusConverter:
    """Orchestrates document conversion and artifact management.

    ``caption_service`` (a marker BaseService or None) drives the custom
    captioning of *extracted* figures. ``describe_images`` keeps marker's
    native ``LLMImageDescriptionProcessor`` in the pipeline; when False it is
    dropped so no figure descriptions are produced unless explicitly asked.
    """

    def __init__(
        self,
        config_dict: dict[str, Any],
        caption_service: Any | None,
        describe_images: bool,
        device: str,
    ):
        self.config_parser = ConfigParser(config_dict)
        self.artifact_dict = create_model_dict(device=device)
        self.caption_service = caption_service

        # get_processors() returns None unless --processors is given, and
        # PdfConverter expects import strings, so to drop the figure-description
        # step we rebuild marker's default list (as strings) minus that one.
        processor_list = None
        if not describe_images:
            processor_list = [
                f"{cls.__module__}.{cls.__name__}"
                for cls in PdfConverter.default_processors
                if cls.__name__ != "LLMImageDescriptionProcessor"
            ]

        self.converter = PdfConverter(
            config=self.config_parser.generate_config_dict(),
            artifact_dict=self.artifact_dict,
            processor_list=processor_list,
            renderer=self.config_parser.get_renderer(),
            llm_service=self.config_parser.get_llm_service(),
        )

    def process_file(self, file_path: Path, output_dir: Path, name: str) -> None:
        """Converts a single file into ``{output_dir}/{name}/{name}.md``."""
        doc_out_dir = output_dir / name
        doc_out_dir.mkdir(parents=True, exist_ok=True)

        rendered = self.converter(str(file_path))

        md_content = post_process_markdown(rendered.markdown)
        images_dict = rendered.images

        artifact_map = {}
        if images_dict:
            image_dir = doc_out_dir / "images"
            image_dir.mkdir(exist_ok=True)

            for img_name, img_obj in images_dict.items():
                img_obj.save(image_dir / img_name)

                if self.caption_service is None:
                    continue

                caption = caption_image(self.caption_service, img_obj)
                artifact_map[img_name] = caption

                md_img_link = f"![{img_name}]({img_name})"
                if md_img_link in md_content:
                    md_content = md_content.replace(
                        md_img_link,
                        f"![{img_name}](images/{img_name})\n\n*Caption: {caption}*\n",
                    )

            if artifact_map:
                map_path = doc_out_dir / "artifact_mapping.json"
                map_path.write_text(
                    json.dumps(artifact_map, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        md_path = doc_out_dir / f"{name}.md"
        with open(md_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(md_content)


# ---------------------------------------------------------------------------
# Resolution reporting
# ---------------------------------------------------------------------------


def prompt_for_pdf(res: Resolution) -> Path | None:
    """Interactively asks the user for a local PDF path for one entry."""
    title = res.entry.get("title", "").strip("{} ")
    doi = res.entry.get("doi", "-")
    console.print(f"\n[yellow]Unresolved:[/] {res.citekey}")
    console.print(f"  Title: {title[:80]}")
    console.print(f"  DOI:   {doi}")
    while True:
        raw = console.input("  PDF path (Enter to skip): ").strip()
        if not raw:
            return None
        candidate = Path(os.path.expanduser(raw))
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            return candidate
        console.print("  [red]Not a readable .pdf, try again (or Enter to skip).[/]")


def print_plan(results: list[Resolution]) -> None:
    """Renders the resolution plan as a table."""
    table = Table(title="PDF resolution plan")
    table.add_column("Citekey", style="cyan", no_wrap=True)
    table.add_column("Source")
    table.add_column("Path / status")
    for r in results:
        status = str(r.path) if r.path else "[red]unresolved[/]"
        table.add_row(r.citekey, r.source, status)
    console.print(table)


def write_metadata(doc_out_dir: Path, res: Resolution) -> None:
    """Writes citation provenance next to the Markdown for RAG traceability."""
    entry = res.entry
    year = entry.get("year") or entry.get("date", "")[:4]
    meta = {
        "citekey": res.citekey,
        "title": entry.get("title", "").strip("{} "),
        "author": entry.get("author", ""),
        "year": year,
        "doi": entry.get("doi", ""),
        "source": res.source,
        "pdf": str(res.path) if res.path else None,
    }
    (doc_out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


@click.command()
@click.argument(
    "bib_path", type=click.Path(exists=True, path_type=Path, dir_okay=False)
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for Markdown.",
)
@click.option(
    "--pdf-dir",
    type=click.Path(path_type=Path),
    default=Path("./downloads"),
    help="Directory for downloaded PDFs and the {citekey}.pdf drop-in convention.",
)
@click.option("--device", type=str, default="cuda", help="Torch device to use.")
@click.option(
    "--cpu-threads",
    type=int,
    default=None,
    help="Torch intra-op CPU threads (default: physical core count). Lower if unstable.",
)
@click.option(
    "--ocr/--no-ocr", default=True, help="Force OCR for high precision extraction."
)
@click.option(
    "--llm-provider",
    type=click.Choice(["none", *PROVIDERS]),
    default="none",
    help="Vision LLM provider for all LLM features (keys/urls read from .env).",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="Override the provider's default vision model (deployment name for azure).",
)
@click.option(
    "--use-llm/--no-use-llm",
    default=False,
    help="Use marker's in-conversion LLM cleanup (tables/equations/math). Needs a provider.",
)
@click.option(
    "--no-images",
    is_flag=True,
    default=False,
    help="Do not keep extracted figure files (marker extract_images=False).",
)
@click.option(
    "--caption-images",
    is_flag=True,
    default=False,
    help="Describe figures using the chosen provider. Needs a provider.",
)
@click.option(
    "--vision-check/--no-vision-check",
    default=True,
    help="Verify the chosen model is vision-capable before converting.",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Prompt for a local path when a PDF cannot be resolved automatically.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Re-convert entries even if their Markdown already exists.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Resolve PDFs and print the plan without converting.",
)
def main(
    bib_path: Path,
    out_dir: Path,
    pdf_dir: Path,
    device: str,
    cpu_threads: int | None,
    ocr: bool,
    llm_provider: str,
    llm_model: str | None,
    use_llm: bool,
    no_images: bool,
    caption_images: bool,
    vision_check: bool,
    interactive: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Resolve a PDF per BibTeX entry and convert each to Markdown."""
    # Shell environment wins; .env (searched upward from this script) fills gaps.
    load_dotenv()
    os.environ["TORCH_DEVICE"] = device

    # Set before marker is imported: avoid native heap corruption ("corrupted
    # double-linked list") from clashing OpenMP runtimes and forked tokenizers
    # in the torch/surya/opencv stack.
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # A provider is required for any LLM feature.
    if llm_provider == "none" and (use_llm or caption_images):
        raise click.ClickException(
            "--use-llm/--caption-images need a vision provider; set --llm-provider."
        )

    if bib_path.suffix.lower() != ".bib":
        console.log("[Error] Input must be a .bib file.")
        sys.exit(1)

    # --- Phase 1: parse + resolve (offline-first, no GPU yet) ---
    console.log("Parsing BibTeX and resolving PDFs...")
    resolver = PdfResolver(pdf_dir, os.getenv("UNPAYWALL_EMAIL"))
    entries = resolver.parse_bibtex(bib_path)

    if not entries:
        console.log("No entries found in BibTeX file.")
        sys.exit(0)

    results: list[Resolution] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Resolving PDFs...", total=len(entries))
        for entry in entries:
            progress.update(
                task, description=f"Resolving: {entry.get('ID', 'Unknown')}"
            )
            results.append(resolver.resolve(entry))
            progress.advance(task)

    print_plan(results)

    # --- Phase 2: prompt for whatever is left ---
    unresolved = [r for r in results if not r.resolved]
    if unresolved and interactive and sys.stdin.isatty():
        console.log(f"{len(unresolved)} entries unresolved; prompting for paths...")
        for r in unresolved:
            path = prompt_for_pdf(r)
            if path:
                r.path, r.source = path, "manual"
        unresolved = [r for r in results if not r.resolved]
    elif unresolved:
        console.log(
            f"{len(unresolved)} entries unresolved (interactive prompting disabled)."
        )

    resolved = [r for r in results if r.resolved]

    if unresolved:
        report_path = out_dir / "unresolved.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{r.citekey}\t{r.entry.get('doi', '-')}\t{r.entry.get('title', '').strip('{} ')}"
            for r in unresolved
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.log(
            f"[Note] Wrote {len(unresolved)} unresolved entries to {report_path}. "
            f"Drop a {{citekey}}.pdf into {pdf_dir} and re-run to pick them up."
        )

    console.log(f"Resolved {len(resolved)}/{len(entries)} entries.")

    if dry_run:
        console.log("[Dry run] Stopping before conversion.")
        return

    if not resolved:
        console.log("Nothing to convert.")
        return

    # --- Phase 3: the heavy marker pass ---
    load_marker()  # imported now, after TORCH_DEVICE is set, so --device is honored

    import torch  # pyright: ignore[reportMissingImports] # available via marker; imported here to honor --device timing

    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    console.log(
        f"Torch on {device} using {torch.get_num_threads()} intra-op CPU threads."
    )

    keep_images = not no_images
    config: dict[str, Any] = {
        "output_format": "markdown",
        "force_ocr": ocr,
        "format_lines": True,
        "use_llm": use_llm,
        "extract_images": keep_images,
        "pdftext_workers": 1,  # single-process PDF extraction; avoids fork-time crash
    }

    resolved_model: str | None = None
    if llm_provider != "none":
        llm_cfg, resolved_model, base_url = build_llm_config(llm_provider, llm_model)
        config.update(llm_cfg)

        # Verify the model can see images before the expensive run.
        if vision_check and (use_llm or caption_images):
            ok, detail = is_vision_model(llm_provider, resolved_model, base_url)
            if ok is False:
                console.log(
                    f"[Error] '{resolved_model}' is not a vision model ({detail}). "
                    "Pick a vision-capable model or pass --no-vision-check."
                )
                sys.exit(1)
            elif ok is None:
                console.log(
                    f"[Warning] Could not verify '{resolved_model}' is vision-capable "
                    f"({detail}). marker will fail mid-run if it is not."
                )
            else:
                console.log(f"[OK] Vision model verified: {resolved_model} ({detail}).")

    # Decide how figures get described.
    caption_service = None
    describe_images = caption_images
    if caption_images:
        if keep_images:
            # Custom pass: marker won't describe figures it extracts to files.
            caption_service = build_service(llm_provider, config)
            console.log(f"Captioning figures with {llm_provider}:{resolved_model}.")
        elif use_llm:
            # marker-native inline descriptions (needs use_llm + extract_images=False).
            console.log("Using marker-native inline image descriptions.")
        else:
            # No silent auto-enable: warn and skip, per request.
            console.log(
                "[Warning] --caption-images with --no-images needs --use-llm for "
                "marker's native descriptions; captioning will be skipped."
            )
            describe_images = False

    console.log(f"Initializing ConverterEngine on {device}...")
    try:
        converter_engine = CorpusConverter(
            config, caption_service, describe_images, device
        )
    except Exception as e:
        console.log(f"[Error] Failed to initialize Marker: {e}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bib_path, out_dir / bib_path.name)  # keep the bib alongside outputs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=len(resolved))
        for r in resolved:
            if r.path is None:  # guaranteed by the filter; also narrows the type
                continue
            citekey = r.citekey
            doc_out_dir = out_dir / citekey
            doc_out_dir.mkdir(parents=True, exist_ok=True)
            write_metadata(doc_out_dir, r)

            md_path = doc_out_dir / f"{citekey}.md"
            if md_path.exists() and not force:
                progress.update(task, description=f"Skip (exists): {citekey}")
                progress.advance(task)
                continue

            progress.update(task, description=f"Converting: {citekey}")
            try:
                converter_engine.process_file(r.path, out_dir, citekey)
            except Exception as e:
                console.log(f"[Error] Failed processing {citekey}: {e}")
            progress.advance(task)

    console.log("[Done] Literature pipeline execution completed.")


if __name__ == "__main__":
    main()

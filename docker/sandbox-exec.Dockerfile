# Kortny sandbox execution image (HIG-239).
#
# The sandbox-runner spins up throwaway / session containers from THIS image to
# run skill scripts. Those containers run with `--network none` and a read-only
# root filesystem, so there is NO opportunity to `pip install` at runtime — every
# dependency a curated builder skill needs must be baked in here at build time.
#
# Base is the same uv image the runner itself uses, so the Python version and
# tooling match. We pre-install the document-builder Python deps into the system
# environment (so `python scripts/<name>.py` resolves them without uv project
# resolution) and the system libraries WeasyPrint needs to render PDFs.
#
# Keep this list aligned with the skill PROVENANCE.md dependency notes:
#   spreadsheet-builder -> openpyxl
#   deck-builder        -> python-pptx
#   styled-report-pdf   -> weasyprint (+ pango/cairo/gdk-pixbuf system libs);
#                          pymupdf for the post-render pagination analysis loop
#   chart-maker         -> matplotlib (+ pandas)
#   (article-extractor/youtube-transcript -> trafilatura / yt-dlp)
#   (slack-gif-creator  -> Pillow + ffmpeg)
#   shared HTML/templating -> jinja2; optional interactive charts -> plotly
#   doc-editor          -> python-docx
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# System packages:
#   - ffmpeg                : slack-gif-creator video->gif path
#   - WeasyPrint runtime    : libpango / libpangocairo / libcairo / libgdk-pixbuf
#                             / libffi / shared-mime-info / fonts so PDFs render
#                             text correctly inside the network-isolated container
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fonts-dejavu-core \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Python deps installed into the system environment with uv pip (no network at
# container runtime, so they must exist in the image). Pinned floors match what
# the skill scripts were tested against.
RUN uv pip install --system --no-cache \
        openpyxl>=3.1 \
        python-pptx>=1.0 \
        weasyprint>=69 \
        pymupdf>=1.24 \
        pandas>=2.2 \
        matplotlib>=3.9 \
        plotly>=5.22 \
        Pillow>=10.3 \
        trafilatura>=1.10 \
        yt-dlp>=2024.8.6 \
        jinja2>=3.1 \
        python-docx>=1.2

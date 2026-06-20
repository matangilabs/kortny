from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from kortny.tools import SlackFileReadError, SlackFileReadTool


class FakeSlackFilesClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[str] = []

    def files_info(self, *, file: str) -> dict[str, Any]:
        self.calls.append(file)
        return self.response


def test_slack_file_read_extracts_pdf_text(tmp_path: Path) -> None:
    pdf_bytes = make_pdf_bytes(tmp_path / "source.pdf", "Kortny can read PDFs.")
    client = FakeSlackFilesClient(
        file_info(
            file_id="F123",
            name="report.pdf",
            mimetype="application/pdf",
            size=len(pdf_bytes),
            url="https://files.slack.com/files-pri/T123-F123/report.pdf",
        )
    )
    transport = download_transport(pdf_bytes, "application/pdf")

    result = SlackFileReadTool(
        client=client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=transport,
    ).invoke({"file_id": "F123"})

    output_path = Path(result.output["path"])
    assert client.calls == ["F123"]
    assert output_path.exists()
    assert output_path.read_bytes() == pdf_bytes
    assert result.output["filename"] == "report.pdf"
    assert result.output["mime_type"] == "application/pdf"
    assert result.output["size_bytes"] == len(pdf_bytes)
    assert result.output["extraction_supported"] is True
    assert "Kortny can read PDFs" in result.output["extracted_text"]


def test_slack_file_read_passthrough_extracts_csv(tmp_path: Path) -> None:
    csv_bytes = b"ticker,price\nAAPL,200\n"
    client = FakeSlackFilesClient(
        file_info(
            file_id="F456",
            name="tickers.csv",
            mimetype="text/csv",
            size=len(csv_bytes),
            url="https://files.slack.com/files-pri/T123-F456/tickers.csv",
        )
    )

    result = SlackFileReadTool(
        client=client,
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=download_transport(csv_bytes, "text/csv; charset=utf-8"),
    ).invoke({"file_id": "F456"})

    assert result.output["extracted_text"] == "ticker,price\nAAPL,200\n"
    assert result.output["extracted_text_truncated"] is False


def test_slack_file_read_rejects_oversized_file_before_download(
    tmp_path: Path,
) -> None:
    client = FakeSlackFilesClient(
        file_info(
            file_id="F789",
            name="large.pdf",
            mimetype="application/pdf",
            size=11,
            url="https://files.slack.com/files-pri/T123-F789/large.pdf",
        )
    )

    with pytest.raises(SlackFileReadError, match="too large"):
        SlackFileReadTool(
            client=client,
            bot_token="xoxb-test",
            working_dir=tmp_path,
            max_file_size_bytes=10,
            transport=download_transport(b"not reached", "application/pdf"),
        ).invoke({"file_id": "F789"})


def test_slack_file_read_requires_file_id_or_url(tmp_path: Path) -> None:
    client = FakeSlackFilesClient(file_info(file_id="F000"))

    with pytest.raises(ValueError, match="file_id"):
        SlackFileReadTool(
            client=client,
            bot_token="xoxb-test",
            working_dir=tmp_path,
        ).invoke({})


def test_slack_file_read_reports_invalid_file_id_as_recoverable(
    tmp_path: Path,
) -> None:
    result = SlackFileReadTool(
        client=FakeSlackFilesClient(file_info(file_id="unused")),
        bot_token="xoxb-test",
        working_dir=tmp_path,
    ).invoke({"file_id": "1779562391.617439"})

    assert result.output == {
        "file_id": "1779562391.617439",
        "file_url": None,
        "error": {
            "code": "invalid_file_id",
            "message": (
                "slack_file_read file_id must be a Slack file ID like F123ABC, "
                "not a message timestamp or filename"
            ),
            "recoverable": True,
        },
    }


def test_slack_file_read_reports_file_not_found_as_recoverable(
    tmp_path: Path,
) -> None:
    result = SlackFileReadTool(
        client=FakeSlackFilesClient({"ok": False, "error": "file_not_found"}),
        bot_token="xoxb-test",
        working_dir=tmp_path,
    ).invoke({"file_id": "F404"})

    assert result.output == {
        "file_id": "F404",
        "file_url": None,
        "error": {
            "code": "file_not_found",
            "message": "files.info failed: file_not_found",
            "recoverable": True,
        },
    }


def test_slack_file_read_accepts_private_file_url(tmp_path: Path) -> None:
    text_bytes = b"plain text file"

    result = SlackFileReadTool(
        client=FakeSlackFilesClient(file_info(file_id="unused")),
        bot_token="xoxb-test",
        working_dir=tmp_path,
        transport=download_transport(text_bytes, "text/plain"),
    ).invoke(
        {
            "file_url": "https://files.slack.com/files-pri/T123-FURL/plain.txt",
        }
    )

    assert result.output["file_id"] is None
    assert result.output["filename"] == "plain.txt"
    assert result.output["extracted_text"] == "plain text file"


def file_info(
    *,
    file_id: str,
    name: str = "report.pdf",
    mimetype: str = "application/pdf",
    size: int = 1,
    url: str = "https://files.slack.com/files-pri/T123-F123/report.pdf",
) -> dict[str, Any]:
    return {
        "ok": True,
        "file": {
            "id": file_id,
            "name": name,
            "mimetype": mimetype,
            "size": size,
            "url_private_download": url,
        },
    }


def download_transport(content: bytes, content_type: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer xoxb-test"
        return httpx.Response(
            200,
            content=content,
            headers={
                "content-type": content_type,
                "content-length": str(len(content)),
            },
            request=request,
        )

    return httpx.MockTransport(handler)


def make_pdf_bytes(path: Path, text: str) -> bytes:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 720, text)
    pdf.save()
    return path.read_bytes()

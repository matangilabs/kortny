"""Content-addressed extraction cache repository for slack_file_read (HIG-279)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from kortny.db.models import FileExtractionCache
from kortny.tools.slack_file_read import TextExtraction


class FileExtractionCacheRepository:
    """Read/write access to the file_extraction_cache table.

    Takes a SQLAlchemy Session. All operations are synchronous and
    the caller is responsible for commit/rollback.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, content_sha256: str) -> TextExtraction | None:
        """Return a cached TextExtraction on hit, updating last_accessed_at.

        Returns None on cache miss.
        """
        row = self._session.get(FileExtractionCache, content_sha256)
        if row is None:
            return None
        # Bump last_accessed_at so retention sweeps preserve hot entries.
        row.last_accessed_at = datetime.now(UTC)
        return TextExtraction(
            supported=row.extraction_supported,
            text=row.extracted_text,
            truncated=row.truncated,
            backend=row.backend,
            warnings=tuple(row.warnings or []),
        )

    def put(
        self,
        content_sha256: str,
        extraction: TextExtraction,
        *,
        byte_size: int,
        page_count: int | None = None,
    ) -> None:
        """Upsert an extraction result.

        Uses ON CONFLICT DO NOTHING — content hash is immutable, so a
        concurrent writer would store the same data.  last_accessed_at is
        kept fresh by the DO NOTHING's WHERE-less touch: a subsequent get()
        will update it.
        """
        stmt = (
            pg_insert(FileExtractionCache)
            .values(
                content_sha256=content_sha256,
                backend=extraction.backend,
                extraction_supported=extraction.supported,
                extracted_text=extraction.text,
                truncated=extraction.truncated,
                page_count=page_count,
                byte_size=byte_size,
                warnings=list(extraction.warnings),
                created_at=func.now(),
                last_accessed_at=func.now(),
            )
            .on_conflict_do_nothing(index_elements=["content_sha256"])
        )
        self._session.execute(stmt)
        # A future ambient job (kortny.ambient sweep) should DELETE
        # rows WHERE last_accessed_at < now() - interval '90 days'.

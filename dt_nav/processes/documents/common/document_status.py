from typing import Optional

from dt_nav.api import DBConn
from sqlalchemy.orm import Session
from werkzeug.exceptions import NotFound

from .documents_common import DocumentNeedle, get_document_by_needle

__all__ = ["get_document_status"]


def get_document_status(needle: DocumentNeedle, db: Optional[Session] = None):
    with DBConn.ensure_session(db):
        document = get_document_by_needle(needle, db, allow_null=True)
        if document is None:
            raise NotFound(f"Document not found: {needle}")

        return {
            "id": document.id,
            "object_type": document.object_type,
            "system_id": document.system_id,
            "root_id": document.root_id,
            "is_active": document.is_active,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
            "ner_extracted": document.ner_text == document.text,
        }

from typing import List, Optional, Sequence, Union

import sqlalchemy as sa
from dt_nav.api import DBConn
from dt_nav.models import Document
from dt_nav.utils import RecomException
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, load_only

DocumentNeedle = Union[Document, int, str]

DOCUMENT_NEEDLE_PARAMS = {
    "description": "Селектор документа. <br> Если число, то id документа в Навигаторе. Если строка, то вида <тип-документа>:<внешний-id-документа>",
    "example": "rpd:1",
}


__all__ = [
    "get_document_by_needle",
    "get_documents_by_needles",
    "DocumentNeedle",
    "get_roots",
    "DOCUMENT_NEEDLE_PARAMS",
]


def _get_where_by_needle(needle: DocumentNeedle):
    if isinstance(needle, Document):
        raise ValueError(f"This needle is already a document: {needle}")
    elif isinstance(needle, int) or isinstance(needle, str) and needle.isnumeric():
        return sa.and_(Document.id == needle)
    elif isinstance(needle, str):
        split = needle.split(":", 2)
        if len(split) < 2:
            raise RecomException(
                f'Needle should be in the format "<id>" or "<type>:<system-id>", given: {needle}'
            )
        object_type, system_id = split
        return sa.and_(
            Document.system_id == system_id,
            Document.object_type == object_type,
        )
    raise RecomException(
        f'Needle should be in the format "<id>" or "<type>:<system-id>", given: {needle}'
    )


def get_document_by_needle(
    needle: DocumentNeedle,
    db: Optional[Session] = None,
    ensure_root=False,
    allow_null=False,
):
    if allow_null is True:
        try:
            return get_document_by_needle(needle, db, ensure_root=ensure_root)
        except NoResultFound:
            return None

    result = None
    with DBConn.ensure_session(db) as db:
        if isinstance(needle, Document):
            result = needle
        else:
            query = _get_where_by_needle(needle)
            result = db.execute(sa.select(Document).where(query)).scalar_one()
        if result is not None and ensure_root is True and result.root_id is not None:
            result = db.execute(
                sa.select(Document).where(Document.id == result.root_id)
            ).scalar_one()
    return result


def get_documents_by_needles(
    needles: Sequence[DocumentNeedle],
    db: Optional[Session] = None,
    fields: Optional[List[str]] = None,
) -> List[Union[Document, None]]:
    results = []
    queries = []
    for needle in needles:
        if not isinstance(needle, Document):
            queries.append(_get_where_by_needle(needle))
    documents_by_needle_ids, documents_by_needle_system_ids = {}, {}
    if len(queries) > 0:
        with DBConn.ensure_session(db) as db:
            query = sa.select(Document).where(sa.or_(*queries))
            if fields:
                query.options(
                    load_only(*[getattr(Document, f) for f in fields], raiseload=True)
                )
            documents = db.execute(query).scalars().all()
        documents_by_needle_ids = {str(d.id): d for d in documents}
        documents_by_needle_system_ids = {
            f"{d.object_type}:{d.system_id}": d for d in documents
        }

    for needle in needles:
        if isinstance(needle, Document):
            results.append(needle)
        else:
            results.append(
                documents_by_needle_ids.get(str(needle), None)
                or documents_by_needle_system_ids.get(str(needle), None)
            )
    return results


def get_roots(document_ids: List[int], db: Optional[Session] = None):
    with DBConn.ensure_session(db) as db:
        root_data = db.execute(
            sa.select(Document.root_id, Document.id).where(
                Document.id.in_(document_ids)
            )
        ).all()
    root_ids = [r.root_id or r.id for r in root_data]
    root_mapping = {r.root_id: r.id for r in root_data if r.root_id is not None}
    return root_ids, root_mapping

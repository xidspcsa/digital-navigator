import logging
from typing import Optional, Sequence

import dramatiq
import sqlalchemy as sa
from dt_nav.api.db import DBConn
from dt_nav.models import Document, DocumentKeyword, DocumentKeywordStatus, Keyword
from dt_nav.processes.documents.common import DocumentNeedle, get_document_by_needle
from dt_nav.tasks import broker
from dt_nav.utils import JsonlDatum, unique_values
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tqdm import tqdm

from .extract import extract_entities, extract_entities_many
from .jsonl_common import (
    JsonlDatumStatus,
    add_rejected_entities_from_source,
    filter_rejected_entities,
    fix_kw,
    merge_jsonl_with_status,
)

__all__ = [
    "get_saved_jsonl",
    "extract_entities_for_document",
    "extract_entities_for_document_type",
]


def get_saved_jsonl(
    needle: DocumentNeedle, db: Optional[Session] = None, filter_rejected=False
) -> JsonlDatumStatus:
    """Get saved entities for a document.

    Parameters
    ----------
    needle : DocumentNeedle

    db : Optional[Session]

    filter_rejected : bool
        If True, filter out entities rejected by the user

    Returns
    -------
    JsonlDatumStatus

    """
    with DBConn.ensure_session(db) as db:
        document = get_document_by_needle(needle, db)
        if document.ner_text is None:
            return {"text": document.text, "entities": [], "status": {}}
        data = db.execute(
            sa.select(DocumentKeyword, Keyword)
            .join(Keyword)
            .where(DocumentKeyword.document_id == document.id)
        ).all()
    saved_entities = []
    status = {}
    for document_keyword, keyword in data:
        if document_keyword.meta is None:
            logging.warn(f"Something is wrong in {document_keyword}")
            continue
        entities = document_keyword.meta.get("indices", [])
        status[keyword.value] = document_keyword.status
        for start, end in entities:
            saved_entities.append((start, end, keyword.type))

    saved_entities = sorted(saved_entities, key=lambda e: e[0])
    datum = {
        "text": document.ner_text,
        "entities": saved_entities,
        "status": status,
    }
    if filter_rejected:
        datum = filter_rejected_entities(datum)
    return datum


def _update_keywords(
    data: JsonlDatum, update_classes: bool, db: Session
) -> Sequence[Keyword]:
    """Assert that all keywords from data exist in the database.

    Parameters
    ----------
    data : JsonlDatum

    update_classes : bool
        If True, update keyword classes
    db : Session


    Returns
    -------
    Sequence[Keyword]
        List of all keywords found in data

    """
    target_keywords_values = []
    for start, end, class_ in data["entities"]:
        target_keywords_values.append(
            {"value": fix_kw(data["text"][start:end]), "type": class_}
        )

    target_keywords_values = unique_values(target_keywords_values, "value")

    if len(target_keywords_values) == 0:
        return []

    insert_stmt = pg_insert(Keyword)

    set_block = {
        "updated_at": sa.func.now(),
    }
    if update_classes:
        set_block["type"] = insert_stmt.excluded.type
    upsert_stmt = (
        insert_stmt.values(target_keywords_values).on_conflict_do_update(
            constraint="keyword_value_key", set_=set_block
        )
    ).returning(Keyword)
    res = db.execute(upsert_stmt).scalars().all()
    return res


def save_jsonl_for_document(
    needle: DocumentNeedle,
    datum: JsonlDatumStatus,
    is_user=False,
    db: Optional[Session] = None,
):
    """Save JSONL NER data for document needle.

    This function merges the stored data for needle with the passed
    datum.

    Parameters
    ----------
    needle : DocumentNeedle

    datum : JsonlDatumStatus

    is_user : bool
        If True, treat datum as if it's created by the user,
        i.e. confirming/addind/rejecting existing entities. Otherwise,
        treat datum as created automatically.
    db : Optional[Session]

    """
    with DBConn.ensure_session(db, commit_if_created=True) as db:
        document = get_document_by_needle(needle, db)

        if is_user:
            assert document.ner_text == datum["text"]
            source_datum = get_saved_jsonl(document, db)
            datum = add_rejected_entities_from_source(datum, source_datum)

        keywords = _update_keywords(datum, update_classes=is_user, db=db)
        keywords_by_value = {kw.value: kw for kw in keywords}

        db.execute(
            sa.delete(DocumentKeyword).where(
                sa.and_(
                    DocumentKeyword.document_id == document.id,
                    # To preserve created_at
                    DocumentKeyword.keyword_id.notin_([kw.id for kw in keywords]),
                )
            )
        )
        document_keywords_by_value = {}

        for start, end, _ in datum["entities"]:
            value = fix_kw(datum["text"][start:end])
            try:
                document_keywords_by_value[value]["meta"]["indices"].append(
                    [start, end]
                )
            except KeyError:
                document_keywords_by_value[value] = {
                    "keyword_id": keywords_by_value[value].id,
                    "document_id": document.id,
                    "status": datum["status"].get(
                        value, DocumentKeywordStatus.EXTRACTED
                    ),
                    "meta": {"indices": [[start, end]]},
                }
        # Document might not be in the session
        db.execute(
            sa.update(Document)
            .where(Document.id == document.id)
            .values(ner_text=document.text)
        )
        if len(document_keywords_by_value) > 0:
            insert_stmt = pg_insert(DocumentKeyword)
            upsert_stmt = insert_stmt.values(
                list(document_keywords_by_value.values())
            ).on_conflict_do_update(
                constraint="document_keyword_pkey",
                set_={
                    "status": insert_stmt.excluded.status,
                    "meta": insert_stmt.excluded.meta,
                    "updated_at": sa.text(
                        """CASE
                            WHEN EXCLUDED.status != document_keyword.status OR EXCLUDED.meta != document_keyword.meta
                            THEN now()
                            ELSE document_keyword.updated_at
                            END"""
                    ),
                },
            )
            db.execute(upsert_stmt)


def _check_document_update(document: Document, update=1, verbose=True):
    if document.ner_text is None:
        if verbose:
            logging.info(
                f"{document}: Extracting entities for the first time (update={update})"
            )
        return True
    elif update == 0 or (document.ner_text == document.text and update == 1):
        if verbose:
            logging.info(
                f"{document}: Not updating extracted entities (update={update})"
            )
        return False
    elif document.ner_text != document.text and update >= 1:
        if verbose:
            logging.info(
                f"{document}: Updating entities due to changed text (update={update})"
            )
        return True
    elif update >= 2:
        if verbose:
            logging.info(
                f"{document}: Updating entities unconditionally (update={update})"
            )
        return True
    else:
        raise ValueError(f"Something went wrong with update={update} and {document}")


@dramatiq.actor(max_retries=0, broker=broker)
def extract_entities_for_document(
    needle: DocumentNeedle, update=1, db: Optional[Session] = None
):
    """Extract or update entities for document

    Parameters
    ----------
    needle : DocumentNeedle

    update : int
        If 0, extract only for the first time. If 1, also extract if
        the document has been updated. If 2, extract unconditionally
    db : Optional[Session]
    """
    with DBConn.ensure_session(db, commit_if_created=True) as db:
        document = get_document_by_needle(needle, db)
        if not _check_document_update(document, update):
            return

        target_datum = extract_entities(document.text)
        source_datum = get_saved_jsonl(document, db)

        datum = merge_jsonl_with_status(target_datum, source_datum)
        save_jsonl_for_document(document, datum, is_user=False, db=db)


def _extract_entities_batch(documents: Document, db: Session):
    target_datums = extract_entities_many([d.text for d in documents])
    for document, target_datum in tqdm(
        zip(documents, target_datums), total=len(documents), desc="Saving"
    ):
        source_datum = get_saved_jsonl(document, db)

        datum = merge_jsonl_with_status(target_datum, source_datum)
        save_jsonl_for_document(document, datum, is_user=False, db=db)


@dramatiq.actor(max_retries=0, broker=broker, time_limit=3600000)
def extract_entities_for_document_type(type_: str, update=1):
    """Extract or update entities for all documents of a given type

    Parameters
    ----------
    type_ : str
        Document type
    update : int
        If 0, extract only for the first time. If 1, also extract if
        the document has been updated. If 2, extract unconditionally
    """
    with DBConn.ensure_session(commit_if_created=True) as db:
        document_ids = (
            db.execute(
                sa.select(Document.id).where(
                    sa.and_(
                        Document.object_type == type_,
                        Document.is_active == True,
                        Document.root_id.is_(None),
                        Document.text.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        documents = db.execute(sa.select(Document).where(Document.id.in_(document_ids)))

        batch = []
        batch_i, i = 0, 0
        total = len(document_ids)
        while True:
            i += 1

            document = None
            result = documents.fetchone()
            if result is not None:
                document = result[0]

            if document is not None:
                if not _check_document_update(document, update, verbose=False):
                    continue
                batch.append(document)

            if len(batch) > 1000 or document is None:
                logging.info(
                    f"Starting batch {batch_i} (size={len(batch)}, i={i}, total={total})"
                )
                _extract_entities_batch(batch, db)
                batch = []
                db.commit()

            if document is None:
                break

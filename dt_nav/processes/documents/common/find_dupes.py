import dramatiq
import pandas as pd
import sqlalchemy as sa
from dt_nav.api import DBConn
from dt_nav.models import Document
from dt_nav.nlp.dupes import clusters_with_tfidf
from dt_nav.tasks import broker
from sqlalchemy import orm

from .documents_common import get_document_by_needle

__all__ = ["mark_dupes", "get_dupes_data"]


def _get_list(text_type, db=None):
    text_list = []
    text_id = []
    document_with_type = db.execute(
        sa.select(Document).where(Document.object_type == text_type)
    ).scalars()

    for document in document_with_type:
        if document.text != None:
            text_list.append(document.text)
        else:
            text_list.append("")
        text_id.append(document.id)
    return text_list, text_id


def _get_root_id(cluster_group):
    if cluster_group.iloc[1] == -1:
        return -1
    else:
        return cluster_group.min()


def _get_dupes_df(text_list, id_list):
    cluster_labels = clusters_with_tfidf(text_list)

    df_res = pd.DataFrame({"id": id_list, "cluster": cluster_labels})

    df_res["root_id"] = df_res.groupby("cluster")["id"].transform(_get_root_id)
    df_res.loc[df_res["cluster"] == -1, "root_id"] = None

    filtered_df = df_res[df_res["root_id"].notnull()]
    filtered_df["root_id"] = filtered_df["root_id"].astype(int)

    return filtered_df


@dramatiq.actor(max_retries=0, broker=broker)
def mark_dupes(text_type: str):

    with DBConn.ensure_session() as db:
        text_list, id_list = _get_list(text_type, db)
        dupes_df = _get_dupes_df(text_list, id_list)

        dupes_values = [
            {"id": d.id, "root_id": d.root_id}
            for d in dupes_df.itertuples(index=False)
            if d.id != d.root_id
        ]

        db.execute(
            sa.update(Document)
            .where(Document.object_type == text_type)
            .values(root_id=None)
        )
        db.execute(sa.update(Document), dupes_values)
        db.commit()


def get_dupes_data(needle, db=None):
    with DBConn.ensure_session(db) as db:
        document = get_document_by_needle(needle, ensure_root=True)
        dupes_data = (
            db.execute(
                sa.select(Document)
                .where(Document.root_id == document.id)
                .order_by(Document.id)
                .options(
                    orm.defer(Document.text, raiseload=True),
                    orm.defer(Document.ner_text, raiseload=True),
                )
            )
            .scalars()
            .all()
        )
    return document, dupes_data

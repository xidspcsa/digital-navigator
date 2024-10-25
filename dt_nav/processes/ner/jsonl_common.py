import json
from typing import Dict, TypedDict

from dt_nav.models.document_keyword import DocumentKeywordStatus
from dt_nav.utils import IdentitySet, NEREntities, find_good_substring_indices
from dt_nav.utils.screw import group_list_by

__all__ = [
    "JsonlDatumStatus",
    "merge_jsonl_with_status",
    "add_rejected_entities_from_source",
    "filter_rejected_entities",
    "fix_kw",
]


def fix_kw(kw: str) -> str:
    """Prepare keyword for saving into the database

    Parameters
    ----------
    kw : str

    Returns
    -------
    str
    """
    return kw.lower().strip()


class JsonlDatumStatus(TypedDict):
    text: str
    entities: NEREntities
    status: Dict[str, DocumentKeywordStatus]


def _update_jsonl_by_text(
    source_datum: JsonlDatumStatus, target_text: str
) -> JsonlDatumStatus:
    """Update source_datum to match target_text.

    This is necessary because entities in source_datum are indices of
    the text. As such, if the text is changed, entities have to be
    updated.

    This function does that just by looking for each word in the new
    text again. So, it is a potentially destructive operation.

    Parameters
    ----------
    source_datum : JsonlDatumStatus
        Original datum with entities
    target_text : str
        New text (has to be different from source_datum["text"]!)

    Returns
    -------
    JsonlDatumStatus
        Updated datum

    """
    entities_classes_count: Dict[str, Dict[str, int]] = {}
    entities_status: Dict[str, DocumentKeywordStatus] = {}
    entities: NEREntities = []

    for start, end, class_ in source_datum["entities"]:
        value = fix_kw(source_datum["text"][start:end])
        classes = entities_classes_count.get(value, {})
        classes[class_] = classes.get(class_, 1)
        entities_classes_count[value] = classes

    for value, classes in entities_classes_count.items():
        class_ = max(classes, key=classes.get)
        starts = find_good_substring_indices(target_text.lower(), value)
        ends = [s + len(value) for s in starts]
        entities.extend([(start, end, class_) for start, end in zip(starts, ends)])

        entities_status[value] = source_datum["status"].get(
            value, DocumentKeywordStatus.EXTRACTED
        )
    entities = sorted(entities, key=lambda e: e[0])
    return {"text": target_text, "entities": entities, "status": entities_status}


_STATUS_PRIORITIES = {
    DocumentKeywordStatus.CONFIRMED: 4,
    DocumentKeywordStatus.REJECTED: 3,
    DocumentKeywordStatus.ADDED: 2,
    DocumentKeywordStatus.EXTRACTED: 1,
}


def _merge_jsonl_entities(
    target_datum: JsonlDatumStatus, source: JsonlDatumStatus
) -> JsonlDatumStatus:
    """Merge entities of target_datum with source_datum.

    target_datum stores newly extracted entities, source_datum stores
    saved ones. Both datums have to have the same text!

    The function performs the operation by two steps:
    1. Updating entities status.
       All entities with the status \"EXTRACTED\" are deleted from source
       Then, for each entity found in both target and source:
       - \"CONFIRMED\" or \"ADDED\" in source becomes \"CONFIRMED\" in
         target
       - \"REJECTED\" in source becomes \"REJECTED\" in target
       Then two lists of entities are merged.
    2. Removing duplicates from the resulting list.

    Parameters
    ----------
    target_datum : JsonlDatumStatus
        Datum with newly extracted entities
    source : JsonlDatumStatus
        Datum with saved entities

    Returns
    -------
    JsonlDatumStatus
        Updated datum
    """
    assert target_datum["text"] == source["text"]
    entities: NEREntities = [*target_datum["entities"]]
    status = {**target_datum["status"]}
    target_entities_by_ents = {json.dumps(e): e for e in target_datum["entities"]}

    for e in source["entities"]:
        value = fix_kw(source["text"][e[0] : e[1]])
        source_status = source["status"].get(value, DocumentKeywordStatus.EXTRACTED)
        if source_status == DocumentKeywordStatus.EXTRACTED:
            continue
        key = json.dumps(e)
        if key in target_entities_by_ents:
            if source_status in (
                DocumentKeywordStatus.CONFIRMED,
                DocumentKeywordStatus.ADDED,
            ):
                status[value] = DocumentKeywordStatus.CONFIRMED
            elif source_status == DocumentKeywordStatus.REJECTED:
                status[value] = DocumentKeywordStatus.REJECTED
            continue
        entities.append(e)

    entities_by_index: Dict[int, NEREntities] = {}

    for e in entities:
        start, end, _ = e
        for i in range(start, end):
            try:
                entities_by_index[i].append(e)
            except KeyError:
                entities_by_index[i] = [e]
    indices = sorted(list(set([*entities_by_index.keys()])))

    result, rejected = IdentitySet(), IdentitySet()
    for i in indices:
        entities_here = entities_by_index[i]
        entities_here = [e for e in entities_here if e not in rejected]
        if len(entities_here) > 1:
            entities_here = sorted(
                entities_here,
                key=lambda e: _STATUS_PRIORITIES.get(
                    status[fix_kw(target_datum["text"][e[0] : e[1]])], 0
                ),
                reverse=True,
            )
        result.add(entities_here[0])
        for e in entities_here[1:]:
            rejected.add(e)
    return {
        "text": target_datum["text"],
        "entities": sorted(list(result), key=lambda e: e[0]),
        "status": status,
    }


def merge_jsonl_with_status(
    target_datum: JsonlDatumStatus,
    source_datum: JsonlDatumStatus,
) -> JsonlDatumStatus:
    """Merge target_datum and source_datum.

    See _merge_jsonl_entities for more.

    Parameters
    ----------
    target_datum : JsonlDatumStatus
        Datum with newly extracted entities
    source_datum : JsonlDatumStatus
        Datum with saved entities

    Returns
    -------
    JsonlDatumStatus
        Updated datum
    """
    if target_datum["text"] != source_datum["text"]:
        source_datum = _update_jsonl_by_text(source_datum, target_datum["text"])
    return _merge_jsonl_entities(target_datum, source_datum)


def add_rejected_entities_from_source(
    target_datum: JsonlDatumStatus, source_datum: JsonlDatumStatus
) -> JsonlDatumStatus:
    """Add entities present in source_datum but missing from target_datum as rejected to source_datum.

    Parameters
    ----------
    target_datum : JsonlDatumStatus
        Datum modified by the user
    source_datum : JsonlDatumStatus
        Original datum

    Returns
    -------
    JsonlDatumStatus

    """
    assert target_datum["text"] == source_datum["text"]

    target_entities_by_value = group_list_by(
        target_datum["entities"], lambda e: fix_kw(target_datum["text"][e[0] : e[1]])
    )
    source_entities_by_value = group_list_by(
        source_datum["entities"], lambda e: fix_kw(source_datum["text"][e[0] : e[1]])
    )

    new_entities = []
    new_status = {}

    for value in source_entities_by_value.keys():
        source_status = source_datum["status"][value]
        if (
            source_status
            in (
                DocumentKeywordStatus.REJECTED,
                DocumentKeywordStatus.EXTRACTED,
                DocumentKeywordStatus.CONFIRMED,
            )
            and value not in target_entities_by_value
        ):
            new_entities.extend(source_entities_by_value[value])
            new_status[value] = DocumentKeywordStatus.REJECTED
    target_datum["entities"].extend(new_entities)
    target_datum["status"].update(new_status)

    target_datum["entities"] = sorted(target_datum["entities"], key=lambda e: e[0])
    return target_datum


def filter_rejected_entities(target_datum: JsonlDatumStatus) -> JsonlDatumStatus:
    """Filter out rejected entities from target_datum.

    Parameters
    ----------
    target_datum : JsonlDatumStatus
        Datum with entities

    Returns
    -------
    JsonlDatumStatus
        Datum with rejected entities removed
    """
    target_datum["entities"] = [
        e
        for e in target_datum["entities"]
        if target_datum["status"][fix_kw(target_datum["text"][e[0] : e[1]])]
        != DocumentKeywordStatus.REJECTED
    ]
    target_datum["status"] = {
        k: v
        for k, v in target_datum["status"].items()
        if v != DocumentKeywordStatus.REJECTED
    }
    return target_datum

import fuzzyset
import networkx as nx
from dt_nav.utils import tqdm_v

__all__ = ["dupes_graph", "dupes_canonical", "dupes_clusters"]


def dupes_graph(strings, cutoff=0.95, min_length=100, verbose=False):
    s = fuzzyset.FuzzySet(gram_size_lower=3)
    indices = {}
    G = nx.Graph()
    for i, string in tqdm_v(enumerate(strings), total=len(strings), verbose=verbose):
        G.add_node(i)

        try:
            old_indices = indices[string]
            for old_index in old_indices:
                G.add_edge(i, old_index, weight=1)
        except KeyError:
            if len(string) >= min_length:
                dupe_candidates = s.get(string)

                if dupe_candidates is not None:
                    for score, old_string in dupe_candidates:
                        if score < cutoff:
                            continue
                        old_indices = indices.get(old_string, [])
                        for old_index in old_indices:
                            G.add_edge(i, old_index, weight=score)

                s.add(string)
        try:
            indices[string].append(i)
        except KeyError:
            indices[string] = [i]

    return G


def dupes_canonical(strings, **kwargs):
    G = dupes_graph(strings, **kwargs)
    mapping = {}
    for i in range(len(strings)):
        adj = G.adj[i]
        if len(adj) == 0:
            continue
        min_idx = min(adj.keys())
        if min_idx < i:
            mapping[i] = min_idx
    return mapping


def dupes_clusters(mapping):
    res = {}
    for idx, min_idx in mapping.items():
        try:
            res[min_idx].append(idx)
        except KeyError:
            res[min_idx] = [idx]
    for min_idx in list(res.keys()):
        res[min_idx].append(min_idx)
    return res


if __name__ == "__main__":
    m = dupes_canonical(
        ["hello", "hello", "hello", "world", "world", "worldd"],
        cutoff=0.55,
        min_length=3,
    )

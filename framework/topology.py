"""Dynamic coalition topology management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

nx: Any = None

try:
    import networkx as _networkx
except ImportError:  # pragma: no cover
    pass
else:
    nx = _networkx


@dataclass
class CoalitionTopologyManager:
    """Maintains coalition graphs and periodic reformation."""

    reform_interval: int = 50
    _graph: Any = field(default=None, init=False, repr=False)
    _coalitions: tuple[tuple[str, ...], ...] = field(default=(), init=False, repr=False)

    def reform(
        self,
        utilities: dict[str, float],
        affinities: dict[tuple[str, str], float],
        *,
        round_idx: int,
        dynamic: bool = True,
    ) -> tuple[tuple[str, ...], ...]:
        if nx is None:
            nodes = tuple(sorted(utilities))
            self._coalitions = (nodes,) if nodes else ()
            return self._coalitions
        if self._graph is None or (dynamic and round_idx % self.reform_interval == 0):
            graph = nx.Graph()
            for name, utility in utilities.items():
                graph.add_node(name, utility=utility)
            for (left, right), weight in affinities.items():
                if left == right:
                    continue
                graph.add_edge(left, right, weight=max(0.0, float(weight)))
            self._graph = graph
            try:
                communities = nx.community.louvain_communities(graph, weight="weight", seed=round_idx)
            except AttributeError:
                communities = nx.community.greedy_modularity_communities(graph, weight="weight")
            self._coalitions = tuple(tuple(sorted(group)) for group in communities if group)
        return self._coalitions

    def modularity(self) -> float:
        if nx is None or self._graph is None or not self._coalitions:
            return 0.0
        communities = [set(group) for group in self._coalitions]
        return float(nx.community.modularity(self._graph, communities, weight="weight"))

    def graph_payload(self) -> dict[str, Any]:
        if nx is None or self._graph is None:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [{"id": node, **self._graph.nodes[node]} for node in self._graph.nodes],
            "edges": [
                {"source": left, "target": right, **attrs}
                for left, right, attrs in self._graph.edges(data=True)
            ],
        }

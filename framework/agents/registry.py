"""Registry and factory helpers for assembling agent lineups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .adversarial import AdversaryAgent, DefenderAgent, RefactoringAgent, WolfpackAdversary
from .base import ForecastingAgent, QLearnedAgent
from .hierarchical import BottomUpAgent, EnsembleAggregatorAgent, TopDownAgent


@dataclass(frozen=True)
class AgentRegistry:
    """Flexible container for variable numbers of agents."""

    forecasters: tuple[ForecastingAgent | BottomUpAgent | TopDownAgent | QLearnedAgent, ...] = ()
    adversaries: tuple[AdversaryAgent | WolfpackAdversary | QLearnedAgent, ...] = ()
    defenders: tuple[DefenderAgent, ...] = ()
    refactorer: RefactoringAgent | None = None
    aggregator: EnsembleAggregatorAgent = field(default_factory=EnsembleAggregatorAgent)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AgentRegistry":
        """Build a registry from an agent list config.

        Args:
            config: Registry configuration containing an ``agents`` list.

        Returns:
            Fully populated registry instance.
        """
        specs = config.get("agents", ("forecaster", "adversary", "defender"))

        forecasters: list[ForecastingAgent | BottomUpAgent | TopDownAgent | QLearnedAgent] = []
        adversaries: list[AdversaryAgent | WolfpackAdversary | QLearnedAgent] = []
        defenders: list[DefenderAgent] = []
        refactorer: RefactoringAgent | None = None

        for item in specs:
            if isinstance(item, str):
                agent_type = item
                name = item
                kwargs: dict[str, Any] = {}
            else:
                agent_type = item.get("type", "forecaster")
                name = item.get("name", agent_type)
                kwargs = dict(item.get("kwargs", {}))

            if agent_type == "qlearned-adversary":
                agent_type = "qlearned"
                kwargs["role"] = "adversary"
            elif agent_type == "qlearned-forecaster":
                agent_type = "qlearned"
                kwargs["role"] = "forecaster"

            role = str(kwargs.pop("role", ""))
            agent = create_agent(agent_type=agent_type, name=name, **kwargs)

            if role == "adversary":
                adversaries.append(agent)
                continue
            if role == "forecaster":
                forecasters.append(agent)
                continue

            if isinstance(agent, DefenderAgent):
                defenders.append(agent)
            elif isinstance(agent, RefactoringAgent):
                refactorer = agent
            elif isinstance(agent, (AdversaryAgent, WolfpackAdversary)):
                adversaries.append(agent)
            elif isinstance(agent, QLearnedAgent) and "adversary" in name:
                adversaries.append(agent)
            else:
                forecasters.append(agent)

        return cls(
            forecasters=tuple(forecasters),
            adversaries=tuple(adversaries),
            defenders=tuple(defenders),
            refactorer=refactorer,
            aggregator=EnsembleAggregatorAgent(mode=str(config.get("aggregator_mode", "equal"))),
        )


def create_agent(agent_type: str, name: str, **kwargs: Any) -> Any:
    """Create an agent instance from a short type string."""
    if agent_type == "forecaster":
        return ForecastingAgent(name=name)
    if agent_type == "adversary":
        return AdversaryAgent(
            name=name,
            aggressiveness=float(kwargs.get("aggressiveness", 1.0)),
            attack_cost=float(kwargs.get("attack_cost", 0.0)),
        )
    if agent_type == "defender":
        return DefenderAgent(name=name)
    if agent_type == "refactor":
        return RefactoringAgent(name=name)
    if agent_type == "bottom_up":
        return BottomUpAgent(name=name, segment_weight=float(kwargs.get("segment_weight", 0.3)))
    if agent_type == "top_down":
        return TopDownAgent(name=name, macro_sensitivity=float(kwargs.get("macro_sensitivity", 0.2)))
    if agent_type == "wolfpack":
        return WolfpackAdversary(
            name=name,
            aggressiveness=float(kwargs.get("aggressiveness", 1.0)),
            attack_cost=float(kwargs.get("attack_cost", 0.0)),
            correlation_threshold=float(kwargs.get("correlation_threshold", 0.7)),
        )
    if agent_type == "qlearned":
        algorithm = cast(Literal["q", "wolf", "rarl"], str(kwargs.get("algorithm", "q")))
        return QLearnedAgent(
            name=name,
            q_table_path=kwargs.get("q_table_path") or kwargs.get("q_table"),
            algorithm=algorithm,
        )
    raise ValueError(f"Unknown agent_type: {agent_type}")

"""Public agent API for the forecasting game."""

from .adversarial import AdversaryAgent, DefenderAgent, RefactoringAgent, WolfpackAdversary
from .base import ForecastingAgent, QLearnedAgent, SafeAgentExecutor, default_ollama_repl
from .evolutionary import EvolutionaryAgentPopulation, StrategyVariant
from .hierarchical import BottomUpAgent, EnsembleAggregatorAgent, LLMPolicyAgent, TopDownAgent
from .registry import AgentRegistry, create_agent

__all__ = [
    "AdversaryAgent",
    "AgentRegistry",
    "BottomUpAgent",
    "DefenderAgent",
    "EnsembleAggregatorAgent",
    "EvolutionaryAgentPopulation",
    "ForecastingAgent",
    "LLMPolicyAgent",
    "QLearnedAgent",
    "RefactoringAgent",
    "SafeAgentExecutor",
    "StrategyVariant",
    "TopDownAgent",
    "WolfpackAdversary",
    "create_agent",
    "default_ollama_repl",
]

"""Evolutionary strategy variants for adaptive agent populations."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from ..types import AgentAction


@dataclass(frozen=True)
class StrategyVariant:
    """Parameterized strategy candidate used by evolutionary population updates."""

    name: str
    role: str
    proportion: float
    delta_scale: float
    delta_bias: float
    bankroll: float = 1.0
    subgroup: str = "default"


@dataclass(frozen=True)
class EvolutionaryAgentPopulation:
    """Immutable pool of strategy variants updated by replicator dynamics."""

    variants: tuple[StrategyVariant, ...]
    evolution_rate: float = 0.05
    prune_threshold: float = 0.1
    bankruptcy_floor: float = 0.0

    @classmethod
    def bootstrap(
        cls,
        *,
        population_size: int = 20,
        evolution_rate: float = 0.05,
        seed: int = 42,
        roles: tuple[str, ...] = ("forecaster", "adversary", "defender"),
        subgroup_count: int = 2,
    ) -> "EvolutionaryAgentPopulation":
        """Create an initial population of strategy variants.

        Args:
            population_size: Total number of variants to seed.
            evolution_rate: Replicator-dynamics update rate.
            seed: RNG seed for reproducibility.
            roles: Agent roles to include in the population.
            subgroup_count: Number of topology-aware subgroups to distribute variants across.

        Returns:
            Bootstrapped evolutionary population.
        """
        rng = Random(seed)
        variants: list[StrategyVariant] = []
        per_role = max(1, population_size // max(1, len(roles)))
        for role in roles:
            for idx in range(per_role):
                subgroup = f"group_{idx % max(1, subgroup_count)}"
                variants.append(
                    StrategyVariant(
                        name=f"{role}_variant_{idx}",
                        role=role,
                        proportion=1.0 / per_role,
                        delta_scale=max(0.25, rng.uniform(0.6, 1.4)),
                        delta_bias=rng.uniform(-0.15, 0.15),
                        subgroup=subgroup,
                    )
                )
        return cls(variants=tuple(variants), evolution_rate=evolution_rate)

    def variants_for(self, role: str, *, subgroup: str | None = None) -> tuple[StrategyVariant, ...]:
        """Return all variants matching a role and optional subgroup."""
        items = tuple(v for v in self.variants if v.role == role and (subgroup is None or v.subgroup == subgroup))
        if items:
            return items
        return tuple(v for v in self.variants if v.role == role)

    def sample(self, role: str, rng: Random, *, subgroup: str | None = None) -> StrategyVariant | None:
        """Sample a role-specific variant proportionally to its population share."""
        items = self.variants_for(role, subgroup=subgroup)
        if not items:
            return None
        weights = [max(1e-9, v.proportion) for v in items]
        return rng.choices(items, weights=weights, k=1)[0]

    @staticmethod
    def apply_variant(action: AgentAction, variant: StrategyVariant | None) -> AgentAction:
        """Apply a variant's scaling and bias to an action."""
        if variant is None:
            return action
        return AgentAction(
            actor=action.actor,
            delta=(action.delta * variant.delta_scale) + variant.delta_bias,
        )

    def evolve(self, reward_history: dict[str, list[float]]) -> "EvolutionaryAgentPopulation":
        """Update population proportions using normalized reward history."""
        updated: list[StrategyVariant] = []
        for role in sorted({variant.role for variant in self.variants}):
            role_variants = [variant for variant in self.variants if variant.role == role]
            if not role_variants:
                continue
            raw_fitness = np.asarray(
                [float(np.mean(reward_history.get(variant.name, [0.0]))) for variant in role_variants],
                dtype=float,
            )
            if raw_fitness.size == 0:
                continue
            min_fit = float(raw_fitness.min())
            max_fit = float(raw_fitness.max())
            if max_fit - min_fit < 1e-12:
                normalized = np.ones_like(raw_fitness)
            else:
                normalized = (raw_fitness - min_fit) / (max_fit - min_fit)
            base_props = np.asarray([variant.proportion for variant in role_variants], dtype=float)
            centered = normalized - float(normalized.mean())
            next_props = base_props * (1.0 + (self.evolution_rate * centered))
            next_props = np.clip(next_props, 1e-9, None)
            next_props = next_props / max(next_props.sum(), 1e-12)
            survivors: list[StrategyVariant] = []
            for variant, fitness, proportion in zip(role_variants, normalized, next_props):
                edge = (2.0 * fitness) - 1.0
                kelly_fraction = max(-0.95, min(edge, 1.0))
                bankroll = variant.bankroll * max(0.0, 1.0 + kelly_fraction * self.evolution_rate)
                if fitness < self.prune_threshold or bankroll <= self.bankruptcy_floor:
                    continue
                survivors.append(
                    StrategyVariant(
                        name=variant.name,
                        role=variant.role,
                        proportion=float(proportion),
                        delta_scale=variant.delta_scale,
                        delta_bias=variant.delta_bias,
                        bankroll=float(bankroll),
                        subgroup=variant.subgroup,
                    )
                )
            if not survivors:
                best_idx = int(np.argmax(normalized))
                best = role_variants[best_idx]
                survivors = [
                    StrategyVariant(
                        name=best.name,
                        role=best.role,
                        proportion=1.0,
                        delta_scale=best.delta_scale,
                        delta_bias=best.delta_bias,
                        bankroll=max(best.bankroll, 1e-6),
                        subgroup=best.subgroup,
                    )
                ]
            else:
                total = sum(item.proportion for item in survivors) or 1.0
                survivors = [
                    StrategyVariant(
                        name=item.name,
                        role=item.role,
                        proportion=item.proportion / total,
                        delta_scale=item.delta_scale,
                        delta_bias=item.delta_bias,
                        bankroll=item.bankroll,
                        subgroup=item.subgroup,
                    )
                    for item in survivors
                ]
            updated.extend(survivors)
        return EvolutionaryAgentPopulation(
            variants=tuple(updated),
            evolution_rate=self.evolution_rate,
            prune_threshold=self.prune_threshold,
            bankruptcy_floor=self.bankruptcy_floor,
        )

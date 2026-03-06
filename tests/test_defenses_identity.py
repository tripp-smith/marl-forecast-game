from framework.defenses import IdentityDefense, defense_from_name


def test_identity_defense_is_noop() -> None:
    defense = defense_from_name("identity")

    assert isinstance(defense, IdentityDefense)
    assert defense.defend(1.25, -0.75) == 0.0

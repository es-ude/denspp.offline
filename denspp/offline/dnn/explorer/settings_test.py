from copy import deepcopy

import pytest

from .settings import DefaultSettingsExplorer, ExploreStrategySearch, SearchStrategy, SettingsExplorer


@pytest.mark.parametrize(
    "strategy, check",
    [
        (ExploreStrategySearch.random, SearchStrategy.RANDOM_SEARCH),
        (ExploreStrategySearch.evolution, SearchStrategy.EVOLUTIONARY_SEARCH),
    ],
)
def test_settings_strategy(strategy: int, check: str):
    settings: SettingsExplorer = deepcopy(DefaultSettingsExplorer)
    settings.search_strategy = strategy

    assert settings.get_search_strategy == check

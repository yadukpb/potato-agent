"""get_tier's boundaries are the actual leaderboard contract -- pin them
down exactly so a future edit to TIERS can't silently reclassify hardware."""
import pytest

from potato_agent.tiers import get_tier


@pytest.mark.parametrize(
    "tokens_per_sec,expected_name",
    [
        (0.1, "Raw Potato"),
        (1.99, "Raw Potato"),
        (2.0, "Baked Potato"),
        (5.99, "Baked Potato"),
        (6.0, "Buttered Potato"),
        (8.9, "Buttered Potato"),
        (11.99, "Buttered Potato"),
        (12.0, "Golden Fries"),
        (24.99, "Golden Fries"),
        (25.0, "Potato Supreme"),
        (1000.0, "Potato Supreme"),
    ],
)
def test_tier_boundaries(tokens_per_sec, expected_name):
    _, name, _ = get_tier(tokens_per_sec)
    assert name == expected_name


def test_returns_emoji_and_blurb_too():
    emoji, name, blurb = get_tier(8.9)
    assert emoji == "🧈"
    assert name == "Buttered Potato"
    assert "conversation pace" in blurb

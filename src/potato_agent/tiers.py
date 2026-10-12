"""Potato tier system: how weak can your hardware be and still run this?

This is the actual point of the repo (see LEADERBOARD.md), not an afterthought
— the benchmark script certifies a tier from your measured tokens/sec and
prints a row ready to paste into a leaderboard PR.
"""

# (upper bound exclusive, emoji, name, blurb), sorted ascending — first match wins.
TIERS = [
    (2.0, "🥔", "Raw Potato", "it technically works"),
    (6.0, "🍠", "Baked Potato", "usable, grab a coffee while it thinks"),
    (12.0, "🧈", "Buttered Potato", "comfortable conversation pace"),
    (25.0, "🍟", "Golden Fries", "snappy"),
    (float("inf"), "🚀", "Potato Supreme", "overkill, you absolute baller"),
]


def get_tier(tokens_per_sec: float) -> tuple[str, str, str]:
    """Returns (emoji, tier_name, blurb) for a measured tokens/sec figure."""
    for upper_bound, emoji, name, blurb in TIERS:
        if tokens_per_sec < upper_bound:
            return emoji, name, blurb
    return TIERS[-1][1], TIERS[-1][2], TIERS[-1][3]

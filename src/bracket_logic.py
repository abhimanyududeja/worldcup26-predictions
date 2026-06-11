"""
src/13_bracket_logic.py

Official FIFA 2026 World Cup knockout bracket logic.
Maps group standings -> R32 -> R16 -> QF -> SF -> Final using the actual
2026 fixture structure (replaces the simplified Elo-seeded bracket used
in v1's 06b_simulate_adjusted.py).

This module is pure logic. The simulator (14_simulate_v2.py) drives it
with Elo + bivariate Poisson match probabilities.
"""
from typing import Dict, List, Tuple, Optional, Callable

# --- Official 2026 R32 fixtures -----------------------------------------
# Each entry: (match_number, slot_left, slot_right)
# Slot codes:
#   "1X"     = group X winner
#   "2X"     = group X runner-up
#   "3:XYZW" = 3rd-place team from one of groups X,Y,Z,W,...
R32_FIXTURES = [
    (73, "2A",  "2B"),
    (74, "1E",  "3:ABCDF"),
    (75, "1F",  "2C"),
    (76, "1C",  "2F"),
    (77, "1I",  "3:CDFGH"),
    (78, "2E",  "2I"),
    (79, "1A",  "3:CEFHI"),
    (80, "1L",  "3:EHIJK"),
    (81, "1D",  "3:BEFIJ"),
    (82, "1G",  "3:AEHIJ"),
    (83, "2K",  "2L"),
    (84, "1H",  "2J"),
    (85, "1B",  "3:EFGIJ"),
    (86, "1J",  "2H"),
    (87, "1K",  "3:DEIJL"),
    (88, "2D",  "2G"),
]

R16_FIXTURES = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

QF_FIXTURES = [
    (97,  89, 90),
    (98,  93, 94),
    (99,  91, 92),
    (100, 95, 96),
]

SF_FIXTURES = [
    (101, 97, 98),
    (102, 99, 100),
]

FINAL_FIXTURE = (104, 101, 102)  # M103 is bronze, skipped

THIRD_PLACE_SLOTS = {
    match_num: set(slot[2:])
    for match_num, left, right in R32_FIXTURES
    for slot in (left, right)
    if slot.startswith("3:")
}

STAGE = {"r32": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5, "champ": 6}


def assign_third_place_to_slots(qualifying_groups):
    slot_order = sorted(THIRD_PLACE_SLOTS.keys())
    assignment, used = {}, set()

    def backtrack(i):
        if i == len(slot_order):
            return True
        slot = slot_order[i]
        for g in qualifying_groups:
            if g in used or g not in THIRD_PLACE_SLOTS[slot]:
                continue
            assignment[slot] = g
            used.add(g)
            if backtrack(i + 1):
                return True
            used.remove(g)
            del assignment[slot]
        return False

    return assignment if backtrack(0) else None


def resolve_r32_matchups(group_winners, group_runners_up,
                         third_place_teams_by_group, third_place_assignment):
    matchups = {}
    for match_num, left, right in R32_FIXTURES:
        def resolve(slot):
            if slot.startswith("1"):
                return group_winners[slot[1]]
            if slot.startswith("2"):
                return group_runners_up[slot[1]]
            if slot.startswith("3:"):
                g = third_place_assignment[match_num]
                return third_place_teams_by_group[g]
            raise ValueError("Bad slot " + slot)
        matchups[match_num] = (resolve(left), resolve(right))
    return matchups


def play_knockouts(r32_matchups, match_winner_fn):
    winners, deepest = {}, {}

    def play(round_fixtures, get_teams, stage_key):
        for entry in round_fixtures:
            match_num = entry[0]
            a, b = get_teams(entry)
            deepest[a] = max(deepest.get(a, 0), STAGE[stage_key])
            deepest[b] = max(deepest.get(b, 0), STAGE[stage_key])
            winners[match_num] = match_winner_fn(a, b)

    play(R32_FIXTURES, lambda e: r32_matchups[e[0]], "r32")
    play(R16_FIXTURES, lambda e: (winners[e[1]], winners[e[2]]), "r16")
    play(QF_FIXTURES,  lambda e: (winners[e[1]], winners[e[2]]), "qf")
    play(SF_FIXTURES,  lambda e: (winners[e[1]], winners[e[2]]), "sf")

    fnum, l, r = FINAL_FIXTURE
    a, b = winners[l], winners[r]
    deepest[a] = STAGE["final"]
    deepest[b] = STAGE["final"]
    champion = match_winner_fn(a, b)
    winners[fnum] = champion
    deepest[champion] = STAGE["champ"]

    return winners, deepest


if __name__ == "__main__":
    print("Validating src/13_bracket_logic.py")
    print()

    print("[1] Third-place slot assignment")
    qualifying = ["A", "C", "D", "E", "F", "I", "J", "L"]
    assignment = assign_third_place_to_slots(qualifying)
    assert assignment is not None
    assert sorted(assignment.values()) == sorted(qualifying)
    for slot, g in assignment.items():
        assert g in THIRD_PLACE_SLOTS[slot]
    print("    Assignment:", dict(sorted(assignment.items())))
    print("    PASS")
    print()

    print("[2] Spain/France collision (the bug in v1)")
    group_winners = {
        "A": "Mexico", "B": "Switzerland", "C": "Brazil", "D": "United States",
        "E": "Germany", "F": "Netherlands", "G": "Belgium", "H": "Spain",
        "I": "France", "J": "Argentina", "K": "Portugal", "L": "England",
    }
    group_runners_up = {
        "A": "South Korea", "B": "Canada", "C": "Morocco", "D": "Paraguay",
        "E": "Ecuador", "F": "Japan", "G": "Iran", "H": "Uruguay",
        "I": "Norway", "J": "Austria", "K": "Colombia", "L": "Croatia",
    }
    thirds_by_group = {
        "A": "Czech Republic", "C": "Haiti", "D": "Australia", "E": "Ivory Coast",
        "F": "Sweden", "I": "Senegal", "J": "Algeria", "L": "Ghana",
    }
    third_assign = assign_third_place_to_slots(list(thirds_by_group))
    matchups = resolve_r32_matchups(group_winners, group_runners_up,
                                    thirds_by_group, third_assign)
    assert "Spain" in matchups[84]
    assert "France" in matchups[77]
    print("    M77 France's R32:", matchups[77])
    print("    M84 Spain's R32: ", matchups[84])

    rank_order = ["Spain","France","Argentina","England","Brazil","Germany",
                  "Portugal","Netherlands","Belgium","Colombia","Norway","Uruguay",
                  "Ecuador","Mexico","Morocco","Switzerland","United States",
                  "Croatia","Japan","Senegal","Australia","Sweden","Iran",
                  "Paraguay","South Korea","Czech Republic","Canada","Cape Verde",
                  "Saudi Arabia","Iraq","Jordan","Ghana","Panama","DR Congo",
                  "Uzbekistan","Tunisia","New Zealand","Egypt","Algeria","Austria",
                  "Bosnia and Herzegovina","Qatar","Haiti","Scotland","Turkey",
                  "Ivory Coast","Curacao"]
    rank = {t: i for i, t in enumerate(rank_order)}
    def higher_rank_wins(a, b):
        return a if rank.get(a, 999) < rank.get(b, 999) else b

    winners, deepest = play_knockouts(matchups, higher_rank_wins)
    assert winners[97] == "France", "M97 winner should be France"
    assert winners[98] == "Spain",  "M98 winner should be Spain"
    print("    SF1 (M101):", winners[97], "vs", winners[98],
          "- SAME HALF, can't both reach Final")
    print("    PASS")
    print()

    print("[3] Argentina/Portugal collision in QF M100")
    assert winners[95] == "Argentina"
    assert winners[96] == "Portugal"
    print("    QF (M100):", winners[95], "vs", winners[96],
          "- meet in QF as fans predicted")
    print("    PASS")
    print()

    print("=" * 56)
    print("Bracket logic validated. Phase 1 complete.")
    print("Champion in this deterministic test:", winners[104])

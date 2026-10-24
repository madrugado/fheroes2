"""Actor-critic advantages of whole-game GRPO (strategy_game_grpo.day_advantages / group_records with critic values)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import strategy_game_grpo  # noqa: E402


def test_lambda_one_credits_what_the_game_gained_after_the_day():
    rewards = [1.0, -1.0]
    values = [[0.0, 0.5, 0.8], [0.0, -0.5, -0.8]]
    adv = strategy_game_grpo.day_advantages(rewards, values, 1.0)
    # centred: reward +1 / -1; day means 0 -> R~ - V~_d
    assert adv[0] == pytest.approx([1.0, 0.5, 0.2]) and adv[1] == pytest.approx([-1.0, -0.5, -0.2])


def test_lambda_zero_is_one_days_change_and_the_last_day_meets_the_reward():
    rewards = [1.0, -1.0]
    values = [[0.0, 0.5, 0.8], [0.0, -0.5, -0.8]]
    adv = strategy_game_grpo.day_advantages(rewards, values, 0.0)
    assert adv[0] == pytest.approx([0.5, 0.3, 0.2]) and adv[1] == pytest.approx([-0.5, -0.3, -0.2])


def test_values_are_centred_per_day_over_the_games_that_reached_it():
    # Both games equally ahead of the critic (it is biased): no advantage from the bias; a shorter game counts where it exists.
    adv = strategy_game_grpo.day_advantages([0.0, 0.0], [[1.0, 1.0], [1.0]], 1.0)
    assert adv == [pytest.approx([0.0, 0.0]), pytest.approx([0.0])]


def test_group_records_use_the_day_of_each_answer():
    def game(reward, answers):
        return {"reward": reward, "days": [{"t": 1}, {"t": 2}, {"t": 3}], "answers": answers}

    answer = lambda day: {"kind": "army", "event": {"t": day + 1}, "ctx": day, "answer": 0}  # noqa: E731
    group = {"seed": 1, "map": "m", "color": "Blue", "games": [game(1.0, [answer(0), answer(2)]), game(-1.0, [answer(0)])]}
    values = [[0.0, 0.5, 0.9], [0.0, -0.5, -0.9]]
    records = strategy_game_grpo.group_records(group, 0.0, values, 1.0)
    # spread 1, eps 0: game 0 day 0 -> +1, day 2 -> +0.1; game 1 day 0 -> -1
    assert [r["adv"][0] for r in records] == pytest.approx([1.0, 0.1, -1.0])
    plain = strategy_game_grpo.group_records(group, 0.0)
    assert [r["adv"][0] for r in plain] == pytest.approx([1.0, 1.0, -1.0])


def test_game_day_values_give_one_value_per_day_in_hero_units():
    torch = pytest.importorskip("torch")
    from transformer_model import AzBattleTransformer

    torch.manual_seed(0)
    model = AzBattleTransformer().eval()
    model.config["window"] = 512
    games = [{"days": [{"t": 1, "p": "Blue"}, {"t": 2, "p": "Blue"}], "answers": []}, {"days": [{"t": 1, "p": "Blue"}], "answers": []}]
    one = strategy_game_grpo.game_day_values(model, games, [], 1.0)
    three = strategy_game_grpo.game_day_values(model, games, [], 3.0)
    assert [len(v) for v in one] == [2, 1]
    assert three[0][1] == pytest.approx(3 * one[0][1], abs=1e-5)
    assert one[0][0] == pytest.approx(one[1][0], abs=1e-5)  # the same first day

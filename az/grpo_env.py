"""fheroes2 as a multi-turn environment for TRL's GRPOTrainer (`environment_factory`).

The world is a seeded fheroes2 game in the autonomous playtest mode; the agent is an LLM that plays
ONE color on the strategic layer (src/fheroes2/ai/ai_decision.cpp): hero targets, castle building,
hero hiring and the army budget. Everything else — movement, battles, economy mechanics and all
other players — stays with the built-in AI.

Episode (= one GRPO completion):
  reset(seed, map, days, color, ...)  starts the engine and runs it to the first query of the agent's
                                      color; the query text is appended to the prompt;
  choose(option)                      the only tool: answers the pending query, runs the engine to
                                      the next one and returns it (or the final result);
  get_reward()                        finishes the game with built-in answers if the model stopped
                                      early (or hit max_completion_length) and scores it.

Reward = (score(agent game) - score(control game)) / reward_scale - penalties, where the control is
the same seeded game with every decision built-in (cached per setup) and score = army strength +
2000 x castles + 10000 x outcome (strategy_model.label_score). Games are deterministic per (seed,
choices), so the group of generations of one prompt starts from the same state and differs only by
the model's choices — exactly the comparison GRPO's group baseline needs.

Engine processes: one per environment instance, alive while the episode is open. The engine waits
for answers with a blocking read (no timeout), so slow generation is fine; engines run under
`nice` and idle while the model generates. Only public methods other than reset/get_reward become
tools — keep helpers private.
"""

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_names import format_resources, monster_name, object_name, race_name  # noqa: E402
from strategy_policies import ARMY_BUDGETS  # noqa: E402

KINDS = ("target", "build", "hire", "army")
EVENT_KIND = {"decision": "target", "build": "build", "hire": "hire", "army": "army"}

# AutoPlaytest::PlayerState as sent in "game_end" (same table as strategy_bench.OUTCOME_SCORE).
OUTCOME_SCORE = {"0": 1, "1": -1, "2": 0, "3": 0}
OUTCOME_TEXT = {"0": "won", "1": "lost", "2": "time limit reached", "3": "interrupted"}
CASTLE_WEIGHT = 2000.0
OUTCOME_WEIGHT = 10000.0

SYSTEM_PROMPT = (
    "You are the strategic commander of one kingdom in Heroes of Might and Magic II (fheroes2). "
    "The game engine asks you one question at a time: where a hero should go, what a castle should "
    "build, whether to hire a hero, and how much a castle may spend on troops. Battles and movement "
    "are handled automatically. Answer EVERY question by calling the tool `choose` with the number "
    "of an option, never with plain text; the tool returns the next question. For example, to pick "
    "option 2 reply with exactly:\n"
    '<tool_call>\n{"name": "choose", "arguments": {"option": 2}}\n</tool_call>\n'
    "When the tool reports that the game is over, stop. Goal: end the game with more castles and a "
    "stronger army than the default AI would."
)


def player_score(result: dict) -> float:
    return (float(result.get("str", 0)) + CASTLE_WEIGHT * float(result.get("k", 0))
            + OUTCOME_WEIGHT * OUTCOME_SCORE.get(str(result.get("s")), 0))


def default_engine_factory(binary: str = "./fheroes2", niceness: int = 10):
    def factory(map_name: str, days: int, seed: int):
        from strategy_env import StrategyEnv

        return StrategyEnv(binary=binary, map_name=map_name, days=days, playthroughs=1, seed=seed, niceness=niceness)

    return factory


class ControlCache:
    """Scores of the all-built-in control games, keyed by (map, days, seed). Shared by all
    environment instances of a process (thread-safe); computed lazily on first use."""

    def __init__(self):
        self._results: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def get(self, engine_factory, map_name: str, days: int, seed: int) -> dict:
        key = (map_name, days, seed)
        with self._lock:
            if key in self._results:
                return self._results[key]
        engine = engine_factory(map_name, days, seed)
        try:
            summaries = engine.run(lambda _ev: None)  # skip everything = the built-in game
        finally:
            engine.close()
        if not summaries:
            raise RuntimeError(f"control game {key} did not report game_end")
        results = {r["c"]: r for r in summaries[-1].get("results") or []}
        with self._lock:
            self._results[key] = results
        return results


CONTROL_CACHE = ControlCache()
EPISODE_LOG_LOCK = threading.Lock()


class HeroesStrategyEnv:
    """One fheroes2 game per episode; see the module docstring. Constructor arguments are the
    defaults; per-example values come from the dataset row passed to reset(). `max_options` caps only
    the hero target list (sorted by value; the built-in choice is the first) — build and hire lists
    are short and the built-in choice may be anywhere in them."""

    def __init__(self, binary: str = "./fheroes2", niceness: int = 10, map_name: str = "2kings.mp2", days: int = 7,
                 agent_days: int | None = None, kinds: tuple[str, ...] = KINDS, max_options: int = 8,
                 show_advisor: bool = True, reward_scale: float = 1000.0, invalid_penalty: float = 0.1,
                 unanswered_penalty: float = 1.0, engine_factory=None, control_cache: ControlCache | None = None,
                 episode_log: str | None = None):
        self._engine_factory = engine_factory or default_engine_factory(binary, niceness)
        self._control_cache = control_cache or CONTROL_CACHE
        self._defaults = dict(map=map_name, days=days, agent_days=agent_days, kinds=tuple(kinds))
        self._max_options = max_options
        self._show_advisor = show_advisor
        self._reward_scale = reward_scale
        self._invalid_penalty = invalid_penalty
        self._unanswered_penalty = unanswered_penalty
        self._episode_log = episode_log
        self._engine = None
        self._reset_state()

    # ----------------------------------------------------------------------------------------
    # TRL interface

    def reset(self, **kwargs) -> str:
        """Starts a new game. Dataset columns used: seed, map, days, color (None = the first player
        to move), agent_days (the agent decides until this day, the game runs to `days`), kinds."""
        self._close_engine()
        self._reset_state()
        self._map = kwargs.get("map") or self._defaults["map"]
        self._days = int(kwargs.get("days") or self._defaults["days"])
        self._seed = int(kwargs.get("seed") or 0)
        self._color = kwargs.get("color") or None
        agent_days = kwargs.get("agent_days") or self._defaults["agent_days"]
        self._agent_days = int(agent_days) if agent_days else self._days
        kinds = kwargs.get("kinds") or self._defaults["kinds"]
        self._kinds = tuple(kinds.split(",")) if isinstance(kinds, str) else tuple(kinds)

        self._engine = self._engine_factory(self._map, self._days, self._seed)
        observation = self._advance()
        intro = f"Map {self._map}, {self._days}-day game. You play {self._color or 'your kingdom'}.\n\n"
        return "\n\n" + intro + observation

    def choose(self, option: int) -> str:
        """Answer the current question of the game.

        Args:
            option: The number of the chosen option, exactly as listed in the question.

        Returns:
            The next question, or the final result when the game is over.
        """
        if self._pending is None:
            self._invalid += 1
            return "The game is over; there is nothing to answer." if self._over else "There is no pending question."
        try:
            index = int(option)
        except (TypeError, ValueError):
            index = -1
        options = self._pending["options"]
        if not 0 <= index < len(options):
            self._invalid += 1
            return f"Invalid option {option!r}: choose a number from 0 to {len(options) - 1}.\n\n{self._pending['text']}"

        self._send(options[index]["reply"])
        self._answered += 1
        self._log.append({"kind": self._pending["kind"], "t": self._pending["t"], "option": index,
                          "advisor": self._pending["advisor"]})
        self._pending = None
        return self._advance()

    def get_reward(self) -> float:
        """Scores the episode (finishes the game with built-in answers first if needed)."""
        if self._reward is not None:
            return self._reward
        while not self._over:
            if self._pending is not None:
                self._unanswered += 1
                self._pending = None
                self._send({"op": "skip"})
            self._advance()
        self._close_engine()

        agent = (self._result or {}).get(self._color)
        if agent is None:
            self._reward = -self._unanswered_penalty
            self._info = {"error": "no result for the agent's color"}
            self._write_episode()
            return self._reward

        control = self._control_cache.get(self._engine_factory, self._map, self._days, self._seed).get(self._color)
        base = player_score(control) if control is not None else 0.0
        delta = (player_score(agent) - base) / self._reward_scale
        asked = self._answered + self._unanswered
        penalty = self._invalid_penalty * self._invalid
        if asked:
            penalty += self._unanswered_penalty * self._unanswered / asked
        self._reward = delta - penalty
        self._info = {"delta": delta, "penalty": penalty, "answered": self._answered, "unanswered": self._unanswered,
                      "invalid": self._invalid, "agent": agent, "control": control}
        self._write_episode()
        return self._reward

    # ----------------------------------------------------------------------------------------
    # Engine loop

    def _reset_state(self) -> None:
        self._color = None
        self._pending = None
        self._over = False
        self._result = None
        self._reward = None
        self._info = {}
        self._context = None
        self._answered = 0
        self._unanswered = 0
        self._invalid = 0
        self._log: list[dict] = []

    def _advance(self) -> str:
        """Runs the engine until the next query for the agent (returns its text) or the game end
        (returns the final text). Queries of other colors, other kinds, after agent_days, or
        without a real choice are answered with the built-in choice."""
        engine = self._engine
        while True:
            line = engine._reader.read_line(engine.read_timeout) if engine is not None else None
            if line is None:
                self._over = True
                return self._final_text()
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = ev.get("ev")
            if kind == "turn_context":
                if self._color is None:
                    self._color = ev.get("p")
                if ev.get("p") == self._color:
                    self._context = ev
            elif kind in EVENT_KIND:
                query = self._make_query(ev) if self._is_ours(ev) else None
                if query is None:
                    self._send({"op": "skip"})
                    continue
                self._pending = query
                return query["text"]
            elif kind == "game_end":
                self._result = {r["c"]: r for r in ev.get("results") or []}
                self._over = True
                self._close_engine()
                return self._final_text()

    def _is_ours(self, ev: dict) -> bool:
        return (ev.get("p") == self._color and EVENT_KIND[ev["ev"]] in self._kinds
                and int(ev.get("t") or 0) <= self._agent_days)

    def _write_episode(self) -> None:
        if not self._episode_log:
            return
        record = {"map": self._map, "days": self._days, "seed": self._seed, "color": self._color,
                  "reward": self._reward, **self._info, "choices": self._log}
        with EPISODE_LOG_LOCK, open(self._episode_log, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _send(self, obj: dict) -> None:
        if self._engine is None:
            return
        try:
            self._engine._send(obj)
        except (BrokenPipeError, OSError):
            pass  # the engine is gone; the next read hits EOF and ends the episode

    def _close_engine(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    # ----------------------------------------------------------------------------------------
    # Query rendering: every option carries the engine reply it stands for.

    def _make_query(self, ev: dict) -> dict | None:
        kind = EVENT_KIND[ev["ev"]]
        options, advisor, question = getattr(self, f"_options_{kind}")(ev)
        if len(options) < 2:
            return None  # no real choice

        lines = [self._header(ev), question]
        for index, option in enumerate(options):
            mark = "  <- default AI" if self._show_advisor and index == advisor else ""
            lines.append(f"  {index}: {option['label']}{mark}")
        return {"kind": kind, "t": ev.get("t"), "options": options, "advisor": advisor, "text": "\n".join(lines)}

    def _header(self, ev: dict) -> str:
        text = f"Day {ev.get('t')}. Resources: {format_resources(ev.get('res') or (self._context or {}).get('res'))}."
        if self._context is not None and self._context.get("t") == ev.get("t"):
            heroes = self._context.get("heroes") or []
            text += f" At the start of the day: castles {len(self._context.get('castles') or [])}, heroes {len(heroes)}"
            if heroes:
                text += " (army strength " + ", ".join(f"{h.get('str', 0):.0f}" for h in heroes) + ")"
            text += "."
        return text

    def _hero_mp(self, hero_id) -> str:
        for hero in (self._context or {}).get("heroes") or []:
            if hero.get("id") == hero_id:
                return f" ({hero.get('mp', 0)} of {hero.get('mmp', 0)} move points at the start of the day)"
        return ""

    def _options_target(self, ev: dict):
        cands = (ev.get("cands") or [])[: self._max_options]
        options = [{"label": f"{object_name(c.get('obj', 0))}, value {c.get('v', 0):.0f}, path cost {c.get('d', 0)}",
                    "reply": {"op": "pick", "h": ev.get("h"), "i": c["i"]}} for c in cands]
        # The built-in AI takes the top-value candidate (the engine sends them sorted by value).
        return options, 0, f"Hero {ev.get('h')}{self._hero_mp(ev.get('h'))}: where should the hero go?"

    def _options_build(self, ev: dict):
        # The built-in building choice is only known after the fact (build_result), so option 0
        # delegates to it; an explicit "nothing" is not equivalent (the built-in path may still act).
        options = [{"label": "let the default AI decide", "reply": {"op": "skip"}},
                   {"label": "build nothing today (save resources)", "reply": {"op": "build", "castle": ev.get("castle"), "b": 0}}]
        for c in ev.get("cands") or []:  # never capped: the built-in choice may be any of them
            trade = " (needs a marketplace trade)" if c.get("trade") else ""
            options.append({"label": f"{c.get('name', c['b'])}, cost {format_resources(c.get('cost'))}{trade}",
                            "reply": {"op": "build", "castle": ev.get("castle"), "b": c["b"]}})
        defensive = " The castle is under threat." if ev.get("defensive") else ""
        return options, 0, f"Castle at tile {ev.get('castle')} ({race_name(ev.get('race', 0))}).{defensive} What to build?"

    def _options_hire(self, ev: dict):
        options = [{"label": "hire nobody", "reply": {"op": "hire", "castle": -1}}]
        cands = ev.get("cands") or []
        for c in cands:
            options.append({"label": f"{race_name(c.get('race', 0))} hero level {c.get('lvl', 1)}, value {c.get('val', 0):.0f}"
                                     f", in castle at tile {c.get('castle')}",
                            "reply": {"op": "hire", "castle": c.get("castle"), "slot": c.get("slot")}})
        bi = ev.get("bi", -1)
        advisor = bi + 1 if isinstance(bi, int) and 0 <= bi < len(cands) else (0 if bi == -1 else None)
        return options, advisor, f"You have {ev.get('heroes', 0)} hero(es). Hire a new hero (2500 gold)?"

    def _options_army(self, ev: dict):
        offer = ", ".join(f"{o.get('n', 0)} {monster_name(o.get('mon', 0))} (of {o.get('avail', 0)})" for o in ev.get("offer") or [])
        options = [{"label": f"spend up to {pct}% of the treasury" + (" (nothing)" if pct == 0 else ""),
                    "reply": {"op": "army", "castle": ev.get("castle"), "pct": pct}} for pct in ARMY_BUDGETS]
        reason = {"defense": "the castle is threatened", "visit": "a hero visits", "hire": "a hero was just hired"}
        question = (f"Castle at tile {ev.get('castle')}: {reason.get(ev.get('reason'), ev.get('reason'))}. "
                    f"Garrison strength {float(ev.get('garrison') or 0):.0f}, hero army {float(ev.get('hero') or 0):.0f}. "
                    f"Affordable troops: {offer or 'none'}. Troop budget?")
        return options, list(ARMY_BUDGETS).index(100), question

    def _final_text(self) -> str:
        mine = (self._result or {}).get(self._color)
        if mine is None:
            return "The game is over."
        state = OUTCOME_TEXT.get(str(mine.get("s")), "finished")
        return (f"The game is over ({state}). You have {mine.get('k', 0)} castle(s), {mine.get('h', 0)} hero(es), "
                f"army strength {mine.get('str', 0)}, gold {mine.get('g', 0)}. Stop now.")


# --------------------------------------------------------------------------------------------
# Dataset


def make_rows(seeds, maps=("2kings.mp2",), days: int = 7, agent_days: int | None = None, colors=(None,),
              kinds: tuple[str, ...] = KINDS) -> list[dict]:
    """One row per (seed, map, color): the conversational prompt + the reset() arguments."""
    rows = []
    for seed in seeds:
        for map_name in maps:
            for color in colors:
                rows.append({
                    "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": "A new game starts."}],
                    "seed": int(seed), "map": map_name, "days": int(days),
                    "agent_days": int(agent_days) if agent_days else int(days),
                    "color": color or "", "kinds": ",".join(kinds),
                })
    return rows


def make_dataset(*args, **kwargs):
    from datasets import Dataset

    return Dataset.from_list(make_rows(*args, **kwargs))

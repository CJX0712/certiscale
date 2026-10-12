"""Anytime-Valid Budget Allocation (AVBA) -- CertiScale's second novel component.

The gap this closes
-------------------
CITE answers *"when do I stop sampling on THIS question?"*.  Production systems
face a different question: given **K questions and one shared sampling budget
B**, which question deserves the next draw?  Uniform round-robin wastes draws
on questions that were settled after three samples and starves the ones that
need two hundred.

Framed properly this is **budget allocation under anytime-valid stopping** -- a
marriage of best-arm identification with sequential testing.  The prior art we
build on:

* Successive Halving / Sequential Halving -- Karnin, Koren & Somekh (2013),
  JMLR 14(1): optimal-ish budget division for best-arm identification.
* UCB-E -- Audibert & Bubeck (2010), COLT: budgeted pure exploration.
* Gittins index -- Gittins (1979): the canonical optimal allocation for
  discounted reward collection.

None of these transfer directly: their reward is a bounded arm mean, whereas
ours is a *supermartingale wealth* whose scale is unbounded and whose stopping
rule is data dependent.  AVBA keeps the UCB-style optimism principle but drives
it with the **log-wealth growth rate** of the underlying e-process, which is
the quantity CITE actually certificates on.

Three strategies are provided, all with the same interface so the benchmark can
compare them head to head:

``uniform``  round-robin (the baseline every deployment falls back to)
``ucb``      UCB-E style optimism on log-wealth growth
``wga``      **Wealth-Gradient Allocation** (default): rank by the empirical
             growth rate of log-wealth over a sliding window, with an optimism
             bonus that decays as a question accumulates samples.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BudgetAllocator", "AllocationState"]


@dataclass
class AllocationState:
    """Per-question bookkeeping owned by the allocator."""

    n_problems: int
    n_draws: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    done: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    wealth_history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_draws.size == 0:
            self.n_draws = np.zeros(self.n_problems, dtype=np.int64)
        if self.done.size == 0:
            self.done = np.zeros(self.n_problems, dtype=bool)

    @property
    def remaining(self) -> int:
        return int(np.sum(~self.done))

    @property
    def open_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.done)


class BudgetAllocator:
    """Decides *which* questions receive the next batch of samples.

    Parameters
    ----------
    n_problems:
        Number of questions ``K`` competing for the budget.
    budget:
        Total number of samples ``B`` to spend across all questions.
    batch_size:
        How many questions to advance per round (``b``).  With ``b = 1`` the
        allocator is fully sequential; larger ``b`` amortises round overhead.
    strategy:
        ``"uniform"`` | ``"ucb"`` | ``"wga"``.
    window:
        Sliding window (in samples) used to estimate log-wealth growth rate.
    exploration_c:
        Optimism coefficient for the ``"ucb"`` / ``"wga"`` bonuses.
    """

    STRATEGIES = ("uniform", "ucb", "wga")

    def __init__(
        self,
        n_problems: int,
        budget: int,
        batch_size: int = 1,
        strategy: str = "wga",
        window: int = 8,
        exploration_c: float = 1.0,
    ):
        if n_problems <= 0:
            raise ValueError("n_problems must be positive")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.n_problems = int(n_problems)
        self.budget = int(budget)
        self.batch_size = int(batch_size)
        self.strategy = strategy
        self.window = int(window)
        self.exploration_c = float(exploration_c)
        self.state = AllocationState(n_problems=self.n_problems)
        self.spent = 0
        self.round = 0

    # -- main entry point ---------------------------------------------------

    def allocate(self, wealth: np.ndarray | None = None) -> np.ndarray:
        """Return the indices of questions to sample in this round.

        Parameters
        ----------
        wealth:
            Current log-wealth ``log E_t`` per question (``-inf`` is not
            allowed; use a large negative number instead).  Ignored by
            ``"uniform"``.

        Returns
        -------
        Array of question indices, length ``<= batch_size``.
        """
        st = self.state
        remaining_budget = self.budget - self.spent
        if remaining_budget <= 0:
            return np.array([], dtype=np.int64)
        open_idx = st.open_indices
        if open_idx.size == 0:
            return np.array([], dtype=np.int64)
        k = int(min(self.batch_size, remaining_budget))

        if self.strategy == "uniform" or wealth is None:
            # Round-robin: the question with the fewest draws goes first, ties
            # broken by index so the result is deterministic.
            order = open_idx[np.argsort(st.n_draws[open_idx], kind="stable")]
            chosen = order[:k]
        else:
            chosen = self._score_based(open_idx, np.asarray(wealth, dtype=np.float64), k)

        self.round += 1
        return np.asarray(chosen, dtype=np.int64)

    def consume(self, n: int) -> None:
        """Record that ``n`` samples were spent."""
        self.spent += int(n)

    def mark_done(self, idx: int) -> None:
        """A question has been certified (or exhausted); free its budget."""
        self.state.done[int(idx)] = True

    def observe(self, wealth: np.ndarray) -> None:
        """Append a wealth snapshot (used for growth-rate estimation)."""
        self.state.wealth_history.append(np.asarray(wealth, dtype=np.float64).copy())

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.budget or self.state.remaining == 0

    # -- internals ----------------------------------------------------------

    def _score_based(self, open_idx: np.ndarray, wealth: np.ndarray, k: int) -> np.ndarray:
        st = self.state
        if st.wealth_history:
            prev = self._history_at(st.wealth_history, offset=self.window)
        else:
            prev = np.zeros(self.n_problems, dtype=np.float64)

        nk = np.maximum(st.n_draws[open_idx], 1)
        growth = (wealth[open_idx] - prev[open_idx]) / np.minimum(
            st.n_draws[open_idx], self.window
        ).clip(1)

        if self.strategy == "ucb":
            # UCB-E style: optimism decays as 1/sqrt(n) in log-space.
            bonus = self.exploration_c * np.sqrt(np.log(max(self.spent, 2)) / nk)
            score = growth + bonus
        else:  # "wga"
            # Wealth-Gradient Allocation: pure growth rate, with an optimism
            # bonus only for questions too young to have a reliable slope.
            young = st.n_draws[open_idx] < self.window
            bonus = np.where(
                young,
                self.exploration_c * (1.0 - st.n_draws[open_idx] / float(self.window)),
                0.0,
            )
            score = growth + bonus

        # Deterministic tie-breaking: sort by (-score, index).
        order = np.lexsort((open_idx, -score))
        return open_idx[order[:k]]

    @staticmethod
    def _history_at(history: list[np.ndarray], offset: int) -> np.ndarray:
        if not history:
            return np.zeros(0, dtype=np.float64)
        j = max(0, len(history) - int(offset))
        return history[j]

    # -- diagnostics --------------------------------------------------------

    def describe(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "n_problems": self.n_problems,
            "budget": self.budget,
            "batch_size": self.batch_size,
            "spent": self.spent,
            "round": self.round,
            "certified": int(np.sum(self.state.done)),
            "draws_per_problem": self.state.n_draws.tolist(),
        }

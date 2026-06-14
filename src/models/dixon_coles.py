"""
src/models/dixon_coles.py
--------------------------
Dixon-Coles Poisson model for match outcome and goals prediction.

Reference:
    Dixon, M.J. & Coles, S.G. (1997). Modelling Association Football Scores
    and Inefficiencies in the Football Betting Market.

Usage:
    from src.models.dixon_coles import DixonColesModel
    model = DixonColesModel(half_life_days=365)
    model.fit(matches_df)
    pred = model.predict("France", "Argentina")
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

logger = logging.getLogger(__name__)

MAX_GOALS = 8  # Scoreline matrix dimension (0..MAX_GOALS each side)


def _poisson_logpmf_vec(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Vectorized Poisson log-PMF: log(e^{-mu} * mu^k / k!)."""
    return -mu + k * np.log(np.maximum(mu, 1e-10)) - gammaln(k + 1)


class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model with:
    - Per-team attack and defense parameters
    - Home advantage parameter
    - Low-score correction (rho)
    - Time-decay weighting (exponential, configurable half-life)

    Parameters
    ----------
    half_life_days : float
        Half-life for the time-decay weight. Matches played `half_life_days`
        ago receive half the weight of today's matches. Default 365 (1 year).
    """

    def __init__(self, half_life_days: float = 365.0, shrinkage_prior: float = 15.0) -> None:
        """
        Parameters
        ----------
        half_life_days : float
            Half-life for time-decay weights (default 365 = 1 year).
        shrinkage_prior : float
            Equivalent number of "ghost" matches pulling every team's
            attack/defense toward the global mean (0 in log-space = average
            team). Higher values = more regularisation for sparse teams.

            Rule of thumb:
              - 15 (default): kicks in for teams with < ~30 matches;
                              keeps WC-qualified teams from going extreme
              - 8 : lighter; use when all teams have ample match data
              - 0 : disable shrinkage entirely
        """
        self.half_life_days  = half_life_days
        self.shrinkage_prior = shrinkage_prior
        self.params_:      Optional[dict] = None
        self.teams_:       list[str] = []
        self.match_counts_: dict[str, float] = {}  # effective (weighted) match count per team

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _time_weight(self, dates: pd.Series, reference_date: pd.Timestamp) -> np.ndarray:
        """Exponential time-decay weights relative to reference_date."""
        days_ago = (reference_date - dates).dt.days.values.astype(float)
        # clamp negative values (future matches shouldn't appear, but just in case)
        days_ago = np.clip(days_ago, 0, None)
        return np.exp(-np.log(2) * days_ago / self.half_life_days)

    @staticmethod
    def _rho_correction_vec(
        hg: np.ndarray, ag: np.ndarray,
        mu: np.ndarray, nu: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Vectorized Dixon-Coles τ correction for all matches at once.
        Only affects scorelines (0,0), (0,1), (1,0), (1,1).
        """
        tau = np.ones(len(hg))
        tau = np.where((hg == 0) & (ag == 0), 1.0 - mu * nu * rho, tau)
        tau = np.where((hg == 0) & (ag == 1), 1.0 + mu * rho,       tau)
        tau = np.where((hg == 1) & (ag == 0), 1.0 + nu * rho,       tau)
        tau = np.where((hg == 1) & (ag == 1), 1.0 - rho,            tau)
        return np.maximum(tau, 1e-10)

    # Scalar version retained for predict()'s scoreline matrix
    @staticmethod
    def _rho_correction(x: int, y: int, mu: float, nu: float, rho: float) -> float:
        """Dixon-Coles low-score correction factor τ(x,y,μ,ν,ρ)."""
        if x == 0 and y == 0:
            return max(1.0 - mu * nu * rho, 1e-10)
        elif x == 0 and y == 1:
            return max(1.0 + mu * rho, 1e-10)
        elif x == 1 and y == 0:
            return max(1.0 + nu * rho, 1e-10)
        elif x == 1 and y == 1:
            return max(1.0 - rho, 1e-10)
        return 1.0

    def _log_likelihood(
        self,
        params: np.ndarray,
        hi_arr: np.ndarray,   # pre-built home team index array
        ai_arr: np.ndarray,   # pre-built away team index array
        hg_arr: np.ndarray,   # home goals array
        ag_arr: np.ndarray,   # away goals array
        weights: np.ndarray,
    ) -> float:
        """
        Vectorized negative log-likelihood — no Python loops over matches.

        All match data is passed as pre-built integer NumPy arrays so the
        optimizer can call this thousands of times with minimal overhead.
        """
        n_teams = len(self.teams_)
        attack  = params[:n_teams]
        defense = params[n_teams: 2 * n_teams]
        home_adv = params[-2]
        rho      = params[-1]

        # Expected goals: fully vectorized (one value per match)
        mu = np.exp(attack[hi_arr] + defense[ai_arr] + home_adv)  # (N,)
        nu = np.exp(attack[ai_arr] + defense[hi_arr])              # (N,)

        # Poisson log-PMF for all matches at once
        log_p_home = _poisson_logpmf_vec(hg_arr, mu)
        log_p_away = _poisson_logpmf_vec(ag_arr, nu)

        # Vectorized rho correction
        tau = self._rho_correction_vec(hg_arr, ag_arr, mu, nu, rho)
        log_tau = np.log(tau)

        ll = np.dot(weights, log_p_home + log_p_away + log_tau)
        return -float(ll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        matches: pd.DataFrame,
        reference_date: Optional[str] = None,
        cutoff_years: int = 8,
        min_tournament_weight: float = 0.0,
    ) -> "DixonColesModel":
        """
        Fit the Dixon-Coles model to historical match data.

        Parameters
        ----------
        matches : pd.DataFrame
            Cleaned DataFrame (from loader.py).
        reference_date : str or None
            ISO date string for the time-decay reference point.
            Defaults to the last match date in the dataset.
        cutoff_years : int
            Only use matches from the last N years. Drastically reduces the
            parameter space (fewer teams, fewer rows) and speeds up fitting.
            Default 8. Set to 0 to use all history.
        min_tournament_weight : float
            Drop matches below this tournament weight (see loader.py).
            E.g. 0.6 drops most friendlies. Default 0.0 (keep all).

        Returns
        -------
        self
        """
        matches = matches.dropna(subset=["home_score", "away_score"]).copy()
        matches = matches.sort_values("date")

        if reference_date is None:
            ref = matches["date"].max()
        else:
            ref = pd.Timestamp(reference_date)

        # --- Speed optimisation: filter to recent, relevant matches ---
        if cutoff_years > 0:
            cutoff = ref - pd.DateOffset(years=cutoff_years)
            matches = matches[matches["date"] >= cutoff]

        if min_tournament_weight > 0.0 and "tournament_weight" in matches.columns:
            matches = matches[matches["tournament_weight"] >= min_tournament_weight]

        if matches.empty:
            raise ValueError("No matches remain after filtering. Relax cutoff_years or min_tournament_weight.")


        self.teams_ = sorted(
            set(matches["home_team"]) | set(matches["away_team"])
        )
        n_teams = len(self.teams_)
        team_idx = {t: i for i, t in enumerate(self.teams_)}

        # --- Pre-build integer arrays ONCE (outside the optimiser loop) ---
        # Drop any rows whose teams aren't in the index (shouldn't happen, but safe)
        valid_mask = (
            matches["home_team"].isin(team_idx)
            & matches["away_team"].isin(team_idx)
        )
        matches = matches[valid_mask]

        hi_arr = matches["home_team"].map(team_idx).to_numpy(dtype=np.int32)
        ai_arr = matches["away_team"].map(team_idx).to_numpy(dtype=np.int32)
        hg_arr = matches["home_score"].to_numpy(dtype=np.float64)
        ag_arr = matches["away_score"].to_numpy(dtype=np.float64)
        weights = self._time_weight(matches["date"], ref)

        # Initial parameter vector:
        # [attack_0, ..., attack_n, defense_0, ..., defense_n, home_adv, rho]
        x0 = np.concatenate([
            np.zeros(n_teams),   # attack (log scale)
            np.zeros(n_teams),   # defense (log scale)
            [0.1],               # home_adv
            [-0.1],              # rho
        ])

        # Constraint: sum of attack params = 0 (identifiability)
        constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n_teams])}]
        bounds = (
            [(-3, 3)] * n_teams   # attack
            + [(-3, 3)] * n_teams  # defense
            + [(-1, 2)]            # home_adv
            + [(-1, 1)]            # rho
        )

        logger.info("Fitting Dixon-Coles model on %d matches, %d teams…", len(matches), n_teams)
        result = minimize(
            self._log_likelihood,
            x0,
            args=(hi_arr, ai_arr, hg_arr, ag_arr, weights),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning("Optimiser did not converge: %s", result.message)

        params = result.x

        # --- Bayesian shrinkage (James-Stein style) -----------------------
        # For each team, calculate its effective (time-weighted) match count.
        # Teams with few matches have their parameters pulled toward 0 (the
        # global mean in log-space, i.e. perfectly average attack/defense).
        #
        #   shrunk_param = fitted_param * n / (n + prior)
        #
        # A team with n=4 matches and prior=8 → 33% of fitted value kept.
        # A team with n=40 matches → 83% kept (minimal shrinkage).
        # ------------------------------------------------------------------
        if self.shrinkage_prior > 0:
            # Effective match count per team = sum of time-weights for their matches
            eff_counts = np.zeros(n_teams)
            for idx, w in zip(np.concatenate([hi_arr, ai_arr]), np.tile(weights, 2)):
                eff_counts[idx] += w

            shrink = eff_counts / (eff_counts + self.shrinkage_prior)  # in [0, 1]

            # Apply to attack and defense vectors in the params array
            params[:n_teams]          *= shrink   # attack
            params[n_teams:2*n_teams] *= shrink   # defense

            shrunken_count = int((shrink < 0.8).sum())
            if shrunken_count:
                logger.info(
                    "Shrinkage applied to %d/%d teams with < %.0f effective matches.",
                    shrunken_count, n_teams,
                    0.8 * self.shrinkage_prior / (1 - 0.8),  # n at which shrink=0.8
                )

            self.match_counts_ = {
                t: round(float(eff_counts[i]), 1)
                for t, i in team_idx.items()
            }
        # ------------------------------------------------------------------

        self.params_ = {
            "attack":   {t: params[i]           for t, i in team_idx.items()},
            "defense":  {t: params[n_teams + i] for t, i in team_idx.items()},
            "home_advantage": float(params[-2]),
            "rho":            float(params[-1]),
        }
        logger.info(
            "Fit complete. home_advantage=%.3f, rho=%.3f",
            self.params_["home_advantage"],
            self.params_["rho"],
        )
        return self

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = False,
        max_goals: int = MAX_GOALS,
    ) -> dict:
        """
        Predict match probabilities.

        Parameters
        ----------
        home_team, away_team : str
        neutral_venue : bool  (if True, home_advantage is not applied)
        max_goals : int

        Returns
        -------
        dict with:
            score_matrix : np.ndarray  (home_goals × away_goals probabilities)
            home_win, draw, away_win : float
            over_2_5, under_2_5 : float
            btts : float  (both teams to score)
            expected_home_goals, expected_away_goals : float
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        p = self.params_
        home_adv = 0.0 if neutral_venue else p["home_advantage"]

        def get_attack(team: str) -> float:
            v = p["attack"].get(team, None)
            if v is None:
                logger.debug("Team '%s' not in Dixon-Coles model; using global-average attack (0.0).", team)
                return 0.0
            return v

        def get_defense(team: str) -> float:
            v = p["defense"].get(team, None)
            if v is None:
                logger.debug("Team '%s' not in Dixon-Coles model; using global-average defense (0.0).", team)
                return 0.0
            return v

        mu = np.exp(get_attack(home_team) + get_defense(away_team) + home_adv)
        nu = np.exp(get_attack(away_team) + get_defense(home_team))

        # Hard cap on expected goals — values above 5 are physically unrealistic
        # and indicate a missing/sparse team problem.  Cap ensures sane scorelines.
        MAX_XG = 4.5
        mu = min(mu, MAX_XG)
        nu = min(nu, MAX_XG)

        # Build scoreline probability matrix — vectorized via outer product
        goals = np.arange(max_goals + 1, dtype=np.float64)
        home_pmf = np.exp(_poisson_logpmf_vec(goals, mu))  # shape (max_goals+1,)
        away_pmf = np.exp(_poisson_logpmf_vec(goals, nu))
        score_matrix = np.outer(home_pmf, away_pmf)         # shape (G+1, G+1)

        # Apply rho correction only to the 2×2 low-score sub-matrix
        rho = p["rho"]
        score_matrix[0, 0] *= max(1.0 - mu * nu * rho, 1e-10)
        score_matrix[0, 1] *= max(1.0 + mu * rho,      1e-10)
        score_matrix[1, 0] *= max(1.0 + nu * rho,      1e-10)
        score_matrix[1, 1] *= max(1.0 - rho,           1e-10)

        # Normalise
        score_matrix /= score_matrix.sum()

        home_win = float(np.tril(score_matrix, -1).sum())
        draw     = float(np.trace(score_matrix))
        away_win = float(np.triu(score_matrix, 1).sum())

        # Total goals distribution via anti-diagonal sums (i + j = total goals)
        # NOTE: np.trace walks main diagonals (col-row = const), NOT anti-diagonals
        # (col+row = const). Must sum manually to get P(total goals = t).
        g = max_goals
        total_goals_prob = np.zeros(2 * g + 1)
        for t in range(2 * g + 1):
            for i in range(max(0, t - g), min(t + 1, g + 1)):
                total_goals_prob[t] += score_matrix[i, t - i]
        over_2_5  = float(total_goals_prob[3:].sum())
        under_2_5 = 1.0 - over_2_5

        btts = float(score_matrix[1:, 1:].sum())

        return {
            "score_matrix": score_matrix,
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over_2_5": over_2_5,
            "under_2_5": under_2_5,
            "btts": btts,
            "expected_home_goals": mu,
            "expected_away_goals": nu,
        }

    def top_scorelines(self, prediction: dict, n: int = 5) -> list[tuple]:
        """Return top-n most likely scorelines from a prediction result."""
        matrix = prediction["score_matrix"]
        max_g = matrix.shape[0] - 1
        entries = [
            (matrix[i, j], i, j)
            for i in range(max_g + 1)
            for j in range(max_g + 1)
        ]
        entries.sort(reverse=True)
        return [(i, j, round(p, 4)) for p, i, j in entries[:n]]

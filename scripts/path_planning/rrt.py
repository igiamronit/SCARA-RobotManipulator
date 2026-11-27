"""Seeded bidirectional RRT-Connect with sampled straight edges.

Distances use joint-range normalization. Limited rotary axes are intervals,
not wraparound coordinates. The caller supplies the collision oracle.
Reference: Kuffner and LaValle, ICRA 2000, doi:10.1109/ROBOT.2000.844730.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


class PlanningFailure(RuntimeError):
    """No path was found within the specified search budget."""


@dataclass
class Result:
    waypoints: np.ndarray
    raw_waypoints: np.ndarray
    iterations: int
    nodes: int
    state_checks: int
    planning_time_s: float
    direct_path_valid: bool


class RRTConnect:
    def __init__(self, lower, upper, valid, resolution, *, seed=0,
                 step_size=.08, max_iterations=3000, shortcut_attempts=80):
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.resolution = np.asarray(resolution, float)
        if (self.lower.ndim != 1 or self.lower.shape != self.upper.shape or
                self.lower.shape != self.resolution.shape or
                not np.isfinite([self.lower, self.upper, self.resolution]).all() or
                np.any(self.upper <= self.lower) or np.any(self.resolution <= 0)):
            raise ValueError("Finite ordered bounds and positive edge resolution are required.")
        if (not np.isfinite(step_size) or step_size <= 0 or
                not isinstance(max_iterations, int) or max_iterations < 1 or
                not isinstance(shortcut_attempts, int) or shortcut_attempts < 0):
            raise ValueError("Invalid RRT search budget or step size.")
        self.scale = self.upper - self.lower
        self.valid = valid
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.shortcut_attempts = shortcut_attempts
        self.checks = 0

    def state_valid(self, q):
        q = np.asarray(q, float)
        if (q.shape != self.lower.shape or not np.isfinite(q).all() or
                np.any(q < self.lower) or np.any(q > self.upper)):
            return False
        self.checks += 1
        return bool(self.valid(q))

    def edge_valid(self, a, b):
        """Check endpoints and increments <= resolution on every joint.

        This is a discrete collision test, not a continuous swept-volume proof.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        if (a.shape != self.lower.shape or b.shape != self.lower.shape or
                not np.isfinite([a, b]).all()):
            return False
        count = max(1, int(np.ceil(np.max(np.abs(b-a)/self.resolution))))
        return all(self.state_valid(a + (b-a)*u) for u in np.linspace(0, 1, count+1))

    def distance(self, a, b):
        return float(np.linalg.norm((a-b)/self.scale))

    def _extend(self, tree, target):
        points, parents = tree
        nearest = int(np.argmin(np.linalg.norm((np.asarray(points)-target)/self.scale, axis=1)))
        source = points[nearest]
        distance = self.distance(source, target)
        if distance < 1e-12:
            return "reached", nearest
        candidate = source + (target-source)*min(1, self.step_size/distance)
        # Preserve the exact target on a connection (and exact goal in the result).
        if distance <= self.step_size:
            candidate = target.copy()
        if not self.edge_valid(source, candidate):
            return "trapped", nearest
        points.append(candidate)
        parents.append(nearest)
        return ("reached" if distance <= self.step_size else "advanced"), len(points)-1

    @staticmethod
    def _trace(tree, index):
        points, parents = tree
        path = []
        while index != -1:
            path.append(points[index])
            index = parents[index]
        return path[::-1]

    def _shortcut(self, path):
        path = [q.copy() for q in path]
        for _ in range(self.shortcut_attempts):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j > i+1 and self.edge_valid(path[i], path[j]):
                path[i+1:j] = []
        return np.asarray(path)

    def plan(self, start, goal):
        began = time.perf_counter()
        self.checks = 0
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not self.state_valid(start):
            raise ValueError("Start configuration is invalid or in collision.")
        if not self.state_valid(goal):
            raise ValueError("Goal configuration is invalid or in collision.")
        direct = self.edge_valid(start, goal)
        if direct:
            path = np.array([start.copy(), goal.copy()])
            return Result(path, path.copy(), 0, 2, self.checks, time.perf_counter()-began, True)
        trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
        for iteration in range(1, self.max_iterations+1):
            active = (iteration-1) % 2
            a, b = trees[active], trees[1-active]
            # Periodic root bias complements uniform samples in the joint box.
            target = b[0][0] if self.rng.random() < .1 else self.rng.uniform(self.lower, self.upper)
            status, ia = self._extend(a, target)
            if status == "trapped":
                continue
            while True:
                status, ib = self._extend(b, a[0][ia])
                if status != "advanced":
                    break
            if status == "reached":
                left, right = self._trace(a, ia), self._trace(b, ib)
                if active:
                    left, right = right, left
                raw = np.asarray(left + right[-2::-1])
                path = self._shortcut(raw)
                return Result(path, raw, iteration, sum(len(t[0]) for t in trees),
                              self.checks, time.perf_counter()-began, False)
        raise PlanningFailure(f"No path found after {self.max_iterations} iterations "
                              f"and {self.checks} state checks.")

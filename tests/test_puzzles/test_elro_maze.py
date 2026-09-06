"""Tests for the Elro Grand Maze exercise."""

import pytest

from puzzles.elro_maze import Solution


@pytest.fixture
def solve():
    return Solution().apply


class TestProvidedExamples:
    """The examples shipped with the problem statement."""

    def test_description_example(self, solve):
        # 4 native walls plus map[3][3], sealed off by map[2][3] and map[3][2].
        maze = [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
        assert solve(maze) == 5

    def test_example_1_start_is_trapped(self, solve):
        # [0, 0] is walled in, so every other tile is unreachable.
        maze = [
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ]
        assert solve(maze) == 15

    def test_example_2_enclosed_tile(self, solve):
        # 4 native walls plus map[2][3], which is enclosed.
        maze = [
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        assert solve(maze) == 5

    def test_example_3_no_enclosed_tiles(self, solve):
        # Nothing is walled off, so only the 3 native walls count.
        maze = [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ]
        assert solve(maze) == 3


class TestEdgeCases:
    """Boundary conditions around the flood fill."""

    def test_single_open_tile(self, solve):
        assert solve([[0]]) == 0

    def test_fully_open_maze(self, solve):
        assert solve([[0] * 5 for _ in range(5)]) == 0

    def test_only_the_start_is_open(self, solve):
        maze = [
            [0, 1],
            [1, 1],
        ]
        assert solve(maze) == 3

    def test_detour_around_a_wall(self, solve):
        # The right column is reachable only by going around the middle wall.
        maze = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        assert solve(maze) == 2

    def test_walls_split_the_maze_in_two(self, solve):
        # The start keeps a 4-tile pocket; the other 7 open tiles form a second
        # region touching it only diagonally, which does not count as a move.
        maze = [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        assert solve(maze) == 5 + 7

    def test_empty_maze(self, solve):
        assert solve([]) == 0
        assert solve([[]]) == 0

    def test_blocked_start_makes_everything_unreachable(self, solve):
        # Outside the stated constraints, but must not report a bogus count.
        assert solve([[1, 0], [0, 0]]) == 4

    def test_input_is_not_mutated(self, solve):
        maze = [
            [0, 1],
            [0, 0],
        ]
        solve(maze)
        assert maze == [[0, 1], [0, 0]]


class TestLargeMaze:
    """The flood fill must stay iterative and linear in the tile count."""

    def test_serpentine_corridor(self, solve):
        # A single 200x200 corridor that snakes through the whole grid: every
        # open tile is reachable, so only the walls are unreachable.
        n = 200
        maze = [[0] * n for _ in range(n)]
        for row in range(1, n, 2):
            blocked = range(1, n) if (row // 2) % 2 == 0 else range(0, n - 1)
            for col in blocked:
                maze[row][col] = 1

        walls = sum(row.count(1) for row in maze)
        assert solve(maze) == walls

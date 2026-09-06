"""
Elro Grand Maze (艾尔罗大迷宫)

An n x n maze is given as a matrix where 0 marks a reachable tile and 1 a
natively blocked one.  The player always starts at [0, 0] (guaranteed to be 0)
and may only move up, down, left or right.

A tile counts as unreachable when it is either natively blocked, or open but
walled off from the start.  Both cases are covered by a single flood fill::

    unreachable = n * n - (tiles reached from [0, 0])

For example::

    [0, 1, 0, 0]
    [0, 0, 0, 0]
    [0, 1, 0, 1]
    [0, 0, 1, 0]

has 4 native walls, and map[3][3] is sealed off by map[2][3] and map[3][2],
so the answer is 5.

Complexity: O(n^2) time, O(n^2) space.
"""

from collections import deque
from typing import Sequence

#
# Note: 类名、方法名、参数名已经指定，请勿修改
#
#
# @param generated_map int整型 二维数组
# @return int整型
#
class Solution:
    def apply(self, generated_map: Sequence[Sequence[int]]) -> int:
        """Return the number of tiles that cannot be reached from [0, 0]."""
        if not generated_map or not generated_map[0]:
            return 0

        n_rows = len(generated_map)
        n_cols = len(generated_map[0])
        total = n_rows * n_cols

        # The constraints guarantee an open start, but a sealed one means
        # nothing at all is reachable.
        if generated_map[0][0] != 0:
            return total

        visited = [[False] * n_cols for _ in range(n_rows)]
        visited[0][0] = True
        reached = 1

        queue = deque([(0, 0)])
        while queue:
            row, col = queue.popleft()
            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if not (0 <= next_row < n_rows and 0 <= next_col < n_cols):
                    continue
                if visited[next_row][next_col] or generated_map[next_row][next_col] != 0:
                    continue
                visited[next_row][next_col] = True
                reached += 1
                queue.append((next_row, next_col))

        return total - reached


if __name__ == "__main__":
    import ast
    import sys

    # Accept the maze as a literal, e.g. [[0,1,1,0],[1,0,0,0],[0,1,0,1],[0,1,1,0]]
    raw = " ".join(sys.argv[1:]) or sys.stdin.read()
    print(Solution().apply(ast.literal_eval(raw.strip())))

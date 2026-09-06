# 艾尔罗大迷宫
#
# 0 是能走的格子,1 是墙,起点固定在 [0,0],只能上下左右走。
# 不可达的格子有两种:本来就是墙的,和虽然是 0 但被墙围住走不过去的。
# 所以从 [0,0] 做一次 BFS,数出能走到多少格,剩下的就都是不可达的:
#     答案 = n * n - 能走到的格子数

from collections import deque


#
# Note: 类名、方法名、参数名已经指定，请勿修改
#
# @param generated_map int整型 二维数组
# @return int整型
#
class Solution:
    def apply(self, generated_map):
        n = len(generated_map)
        if n == 0:
            return 0
        m = len(generated_map[0])

        visited = [[False] * m for _ in range(n)]
        visited[0][0] = True
        count = 1  # 能走到的格子数,起点先算一个

        q = deque()
        q.append((0, 0))
        while q:
            x, y = q.popleft()
            # 上下左右四个方向
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    if not visited[nx][ny] and generated_map[nx][ny] == 0:
                        visited[nx][ny] = True
                        count += 1
                        q.append((nx, ny))

        return n * m - count

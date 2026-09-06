from collections import deque


#
# Note: 类名、方法名、参数名已经指定，请勿修改
#
# @param generated_map int整型 二维数组
# @return int整型
#
class Solution:
    def apply(self, generated_map):
        g = generated_map
        n, m = len(g), len(g[0])
        g[0][0] = 1          # 走过的直接涂成 1,省掉 visited
        q = deque([(0, 0)])
        cnt = 0
        while q:
            x, y = q.popleft()
            cnt += 1
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < n and 0 <= ny < m and g[nx][ny] == 0:
                    g[nx][ny] = 1
                    q.append((nx, ny))
        return n * m - cnt

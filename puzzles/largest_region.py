from collections import deque


#
# Note: 类名、方法名、参数名已经指定，请勿修改
#
# 计算最大区域
# @param arr int整型 二维数组 地图
# @return int整型
#
class Solution:
    def closure(self, arr):
        if not arr or not arr[0]:
            return 0
        n, m = len(arr), len(arr[0])
        mark = [[0] * m for _ in range(n)]   # 每个 1 属于第几块,0 是还没编号
        sz = {}                              # 每块有多大
        t = 1
        for i in range(n):
            for j in range(m):
                if arr[i][j] == 1 and mark[i][j] == 0:
                    mark[i][j] = t
                    q = deque([(i, j)])
                    c = 0
                    while q:
                        x, y = q.popleft()
                        c += 1
                        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                            if 0 <= nx < n and 0 <= ny < m and arr[nx][ny] == 1 and mark[nx][ny] == 0:
                                mark[nx][ny] = t
                                q.append((nx, ny))
                    sz[t] = c
                    t += 1

        ans = max(sz.values()) if sz else 0   # 没有 0 可填的话,答案就是现在最大的那块
        for i in range(n):
            for j in range(m):
                if arr[i][j] == 0:
                    near = set()              # 用 set,不然同一块被数两遍
                    for nx, ny in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if 0 <= nx < n and 0 <= ny < m and mark[nx][ny]:
                            near.add(mark[nx][ny])
                    ans = max(ans, 1 + sum(sz[k] for k in near))
        return ans

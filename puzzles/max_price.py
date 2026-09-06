#
# Note: 类名、方法名、参数名已经指定，请勿修改
#
# @param grid int整型 二维数组
# @return int整型
#
class Solution:
    def maxPrice(self, grid):
        if not grid or not grid[0]:
            return 0
        n, m = len(grid), len(grid[0])
        dp = [[0] * m for _ in range(n)]      # dp[i][j]:走到这格能拿的最高分
        dp[0][0] = grid[0][0]
        for j in range(1, m):                 # 第一行只能一路向右
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, n):                 # 第一列只能一路向下
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[n - 1][m - 1]

# 最高得分的测试

from puzzles.max_price import Solution


def test_example():
    # 1 -> 3 -> 5 -> 2 -> 1 = 12
    assert Solution().maxPrice([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 12


def test_one_cell():
    assert Solution().maxPrice([[7]]) == 7


def test_one_row():
    # 只能一路向右,全都得拿
    assert Solution().maxPrice([[1, 2, 3, 4]]) == 10


def test_one_column():
    assert Solution().maxPrice([[1], [2], [3]]) == 6


def test_greedy_would_fail():
    # 第一步贪心选 9 就掉坑里了,正确答案走下面那条 1+8+9=18
    grid = [
        [1, 9, 0],
        [8, 0, 0],
        [9, 9, 9],
    ]
    assert Solution().maxPrice(grid) == 1 + 8 + 9 + 9 + 9


def test_negative():
    # 分数是负的也得照走,不能中途不动
    assert Solution().maxPrice([[-1, -2], [-3, -4]]) == -7


def test_empty():
    assert Solution().maxPrice([]) == 0

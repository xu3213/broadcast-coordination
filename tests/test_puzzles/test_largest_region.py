# 最大连通区域的测试

from puzzles.largest_region import Solution


def test_example_1():
    # 填哪个 0 都能把两个 1 连起来
    assert Solution().closure([[1, 0], [0, 1]]) == 3


def test_example_2():
    # 全是 1,没得填,直接输出当前面积
    assert Solution().closure([[1, 1], [1, 1]]) == 4


def test_all_zero():
    # 填一个 0,自己就是一块面积 1
    assert Solution().closure([[0, 0], [0, 0]]) == 1


def test_same_block_counted_once():
    # (1,1) 的上边和左边是同一块,不能加两次,答案是 4 不是 7
    maze = [
        [1, 1],
        [1, 0],
    ]
    assert Solution().closure(maze) == 4


def test_join_two_blocks():
    # 中间那个 0 把左右两块 3 格的连起来
    maze = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    assert Solution().closure(maze) == 7


def test_far_away_zero():
    # 右下角那个 0 谁也连不上,老实填在大块旁边
    maze = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    assert Solution().closure(maze) == 5


def test_full_map():
    n = 30
    assert Solution().closure([[1] * n for _ in range(n)]) == 900

# 艾尔罗大迷宫的测试:题目给的例子 + 几个边界情况

from puzzles.elro_maze import Solution


def test_example_in_description():
    # 4 个墙,加上被 map[2][3] 和 map[3][2] 围住的 map[3][3]
    maze = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ]
    assert Solution().apply(maze) == 5


def test_example_1():
    # 起点被困住,除了起点自己以外都到不了
    maze = [
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
    ]
    assert Solution().apply(maze) == 15


def test_example_2():
    # 4 个墙,加上被围住的 map[2][3]
    maze = [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert Solution().apply(maze) == 5


def test_example_3():
    # 没有被围住的空格,只有 3 个墙
    maze = [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]
    assert Solution().apply(maze) == 3


def test_one_cell():
    assert Solution().apply([[0]]) == 0


def test_no_wall():
    maze = [[0] * 5 for _ in range(5)]
    assert Solution().apply(maze) == 0


def test_go_around_the_wall():
    # 右边一列要绕过中间那道墙才能到
    maze = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    assert Solution().apply(maze) == 2


def test_maze_split_in_two():
    # 起点只剩左上角 4 格,另外 7 个空格只跟它斜着挨着,走不过去
    maze = [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    assert Solution().apply(maze) == 5 + 7


def test_big_maze():
    # 300x300 全是空地,BFS 用队列不会爆栈
    n = 300
    maze = [[0] * n for _ in range(n)]
    assert Solution().apply(maze) == 0

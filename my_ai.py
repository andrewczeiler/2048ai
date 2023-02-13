import numpy as np
import random
import time

from game_functions import initialize_game, random_move,\
                            move_down, move_left,\
                            move_right, move_up,\
                            check_for_win, add_new_tile


WEIGHT_MATRIX = [
    [4096, 1024, 256, 64],
    [1024, 256, 64, 16],
    [256, 64, 16, 4],
    [64, 16, 4, 1]
]

'''
WEIGHT_MATRIX = [
    [1073741824, 268435456, 67108864, 16777216],
    [65536, 262144, 1048576, 4194304],
    [16384, 4096, 1024, 256],
    [1, 4, 16, 64]
]
'''




NONE_MATRIX = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

'''
def minmax_player(game,state):
    return minmax_decision(state,game)
def minmax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))
'''

def random_move(board):
    if not isTerminalMax(board):
        action = maxActions(board)
        move = random.randint(0, len(action)-1)
        use_move = action[move]
        board, _, _ = use_move(board)
    game_state = isTerminalMax(board)
    ##if game_state:
        ##print(total_score(board) + " " + str(max_tile(board)))

    return board, not game_state

def total_score(board):
    sum = 0
    for i in range(4):
        for j in range(4):
            sum += board[i][j]
    return str(sum)

def max_tile(board):
    max_tile = 2
    for i in range(4):
        for j in range(4):
            if board[i][j] > max_tile:
                max_tile = board[i][j]
    return max_tile


def minimax_move(board, depth):
    (child, _) = maximize(board, depth)
    return child, not isTerminalMax(child)


def maximize(board, depth):
    (maxChild, maxUtility) = (board, -np.inf)

    if depth == 0 or isTerminalMax(board):
        return maxChild, utility(board)

    depth -= 1

    for move in maxActions(board):
        newBoard = np.copy(board)
        newBoard, _, _ = move(newBoard)
        _, util = minimize(newBoard, depth)
        if util > maxUtility:
            (maxChild, maxUtility) = (newBoard, util)

    return (maxChild, maxUtility)


def minimize(board, depth):
    (minChild, minUtility) = (board, np.inf)

    if depth == 0 or isTerminalMin(board):
        return minChild, utility(board)

    depth -= 1

    for moves in minActions(board):
        newBoard = np.copy(board)
        newBoard = addTile(newBoard, moves[1], moves[2], moves[3])
        _, util = maximize(newBoard, depth)
        if util < minUtility:
            (minChild, minUtility) = (newBoard, util)

    return (minChild, minUtility)


def ab_minimax_move(board, depth):
    (child, _) = ab_maximize(board, depth, -np.inf, np.inf)
    return child, not isTerminalMax(child)


def ab_maximize(board, depth, alpha, beta):
    (maxChild, maxUtility) = (board, -np.inf)

    if depth == 0 or isTerminalMax(board):
        return maxChild, utility(board)

    depth -= 1

    for move in maxActions(board):
        newBoard = np.copy(board)
        newBoard, _, _ = move(newBoard)
        _, util = ab_minimize(newBoard, depth, alpha, beta)
        if util > maxUtility:
            (maxChild, maxUtility) = (newBoard, util)
        if maxUtility >= beta:
            break
        if maxUtility > alpha:
            alpha = maxUtility

    return (maxChild, maxUtility)


def ab_minimize(board, depth, alpha, beta):
    (minChild, minUtility) = (board, np.inf)

    if depth == 0 or isTerminalMin(board):
        return minChild, utility(board)

    depth -= 1

    for moves in minActions(board):
        newBoard = np.copy(board)
        newBoard = addTile(newBoard, moves[1], moves[2], moves[3])
        _, util = ab_maximize(newBoard, depth, alpha, beta)
        if util < minUtility:
            (minChild, minUtility) = (newBoard, util)
        if minUtility <= alpha:
            break
        if minUtility < beta:
            beta = minUtility

    return (minChild, minUtility)


def result(board, action):
    boardCopy = np.copy(board)
    return action(boardCopy)

def maxActions(board):
    actions = []
    board1 = np.copy(board)
    _, move_made, _ = move_up(board1)
    if(move_made):
        actions.append(move_up)
    board2 = np.copy(board)
    _, move_made, _ = move_down(board2)
    if(move_made):
        actions.append(move_down)
    board3 = np.copy(board)
    _, move_made, _ = move_right(board3)
    if(move_made):
        actions.append(move_right)
    board4 = np.copy(board)
    _, move_made, _ = move_left(board4)
    if(move_made):
        actions.append(move_left)
    return actions

def minActions(board):
    actions = []
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                actions.append((board, 2, i, j))
                actions.append((board, 4, i, j))
    return actions


def addTile(board, tileNumber, tilePositionX, tilePositionY):
    board[tilePositionX][tilePositionY] = tileNumber
    return board


def isTerminalMax(board):
    no_moves = 0
    board1 = np.copy(board)
    _, move_made, score = move_up(board1)
    if(not move_made):
        no_moves += 1
    board2 = np.copy(board)
    _, move_made, _ = move_down(board2)
    if(not move_made):
        no_moves += 1
    board3 = np.copy(board)
    _, move_made, _ = move_right(board3)
    if(not move_made):
        no_moves += 1
    board4 = np.copy(board)
    _, move_made, _ = move_left(board4)
    if(not move_made):
        no_moves += 1
    if no_moves == 4:
        return True
    return False


def isTerminalMin(board):
    count = 0
    for i in range(4):
        for j in range(4):
            if board[i][j] != 0:
                count += 1
    if count == 16:
        return True
    return False


def utility(board):
    empty = empty_tiles(board) * 1000
    weighted = weighted_board(board) * 0.03
    max = max_tile(board) * 25
    mono = monotonocity(board) * 100

    return empty + weighted + max + mono


def empty_tiles(board):
    count = 0
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                count += 1
    return count

def weighted_board(board):
    result = 0
    for i in range(4):
        for j in range(4):
            result += board[i][j] * WEIGHT_MATRIX[i][j]
    return result * 10

def max_tile(board):
    max_tile = 2
    for i in range(4):
        for j in range(4):
            if board[i][j] > max_tile:
                max_tile = board[i][j]
    return max_tile


def monotonocity(board):
    result = 0
    for i in range(4):
        for j in range(4):
            number = board[i][j]
            for a in range(4):
                for b in range(4):
                    if not (a == i and b == j):
                        if board[a][b] == number:
                            result += 10 / pow((abs(i-a) + abs(j-b)), 2)
                        if board[a][b] == number/2:
                            result += 2 / pow((abs(i-a) + abs(j-b)), 2)
                        if board[a][b] == number*2:
                            result += 2 / pow((abs(i-a) + abs(j-b)), 2)
    return result



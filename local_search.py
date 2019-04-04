import random
import numpy as np
from copy import deepcopy
import math 

def cost_function(distance_matrix, groups):
    return (
        np.sum([
            np.sum(distance_matrix[group, :][:, group]) / 2
            for group in groups
        ]),
        np.sum([len(group)*(len(group)-1)/2 for group in groups]),
    )

def new_cost(groups, distance_matrix, prev_cost, element, origin, endpoint):
    distance_loss = np.sum(distance_matrix[groups[origin], element])
    distance_gain = np.sum(distance_matrix[groups[endpoint], element])
    removed_edges = len(groups[origin]) - 1
    added_edges = len(groups[endpoint])

    return ((prev_cost[0] + distance_gain - distance_loss), (prev_cost[1] + added_edges - removed_edges))

def get_available_moves(solution):
    moves = [
        (point, origin_idx, endpoint_idx)
        for origin_idx, origin in enumerate(solution) if len(origin) > 1
        for endpoint_idx, endpoint in enumerate(solution) if origin_idx != endpoint_idx
        for point in origin
    ]
    random.shuffle(moves)
    return moves

def apply_move(initial_solution, move):

    solution = deepcopy(initial_solution)
    el, origin, endpoint = move
    solution[origin].remove(el)
    solution[endpoint].append(el)

    return solution

GREEDY = 'GREEDY'
STEEPEST = 'STEEPEST'
MOVE_CHOICE_TACTIC = {
    GREEDY: GREEDY,
    STEEPEST: STEEPEST,
}

def local_search_move(initial_solution, distance_matrix, initial_cost, tactic=MOVE_CHOICE_TACTIC[GREEDY]):
    moves = get_available_moves(initial_solution)
    best_move = None
    best_cost = None
    best_move_gain = 0

    for el, origin, endpoint in moves:
        current_move = el, origin, endpoint
        after_move_cost = new_cost(initial_solution, distance_matrix, initial_cost, el, origin, endpoint)
        total_gain = initial_cost[0] / initial_cost[1] - after_move_cost[0] / after_move_cost[1]
        
        if total_gain > best_move_gain:
            best_move = current_move
            best_move_gain = total_gain
            best_cost = after_move_cost

        if tactic == MOVE_CHOICE_TACTIC[GREEDY] and best_move is not None:
            break

    if best_move is None:
        return None, None

    return best_move, best_cost
    
def local_search(initial_solution, distance_matrix, max_iterations = None, tactic=MOVE_CHOICE_TACTIC[GREEDY]):
    distance_matrix = np.asarray(distance_matrix)
    cost = cost_function(distance_matrix, initial_solution)

    iteration = 0
    solution = initial_solution
    step = True

    step, new_cost = local_search_move(solution, distance_matrix, cost, tactic)

    while step is not None and (max_iterations is None or iteration < max_iterations):
        solution = apply_move(solution, step)
        cost = new_cost
        iteration += 1
        step, new_cost = local_search_move(solution, distance_matrix, cost, tactic)

    return solution


def chunk(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):  
        yield arr[i:i + chunk_size]
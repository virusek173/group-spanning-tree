import random
import numpy as np
from copy import deepcopy

def cost_function(distance_matrix, groups):
    return [
        np.sum(distance_matrix[group, :][:, group]) / 2
        for group in groups
    ]

def move_delta(group_means, distance_matrix, element, groups, origin, endpoint):
    means = list(group_means)
    distance_loss = 2*np.sum(distance_matrix[groups[origin], element])
    distance_gain = 2*np.sum(distance_matrix[groups[endpoint], element])

    origin_n = len(groups[origin])
    dest_n = len(groups[endpoint])
    
    origin_mean = (means[origin] * (origin_n**2) - distance_loss) / ((origin_n - 1)**2)
    endpoint_mean = (means[endpoint] * (dest_n**2) + distance_gain) / ((dest_n + 1)**2)

    origin_delta = origin_mean - means[origin]
    dest_delta = endpoint_mean - means[endpoint]

    return origin_delta, dest_delta




def get_available_moves(solution):
    moves = [
        (point, origin_idx, endpoint_idx)
        for origin_idx, origin in enumerate(solution) if len(origin) > 18
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

def local_search_move(initial_solution, initial_means, distance_matrix, tactic=MOVE_CHOICE_TACTIC[GREEDY]):
    moves = get_available_moves(initial_solution)
    best_move = None
    best_mean_deltas = None
    best_move_gain = 0

    for el, origin, endpoint in moves:
        current_move = el, origin, endpoint
        mean_deltas = move_delta(initial_means, distance_matrix, el, initial_solution, origin, endpoint)
        total_gain = -1 * (mean_deltas[0] + mean_deltas[1])

        if total_gain > best_move_gain:
            best_move = current_move
            best_move_gain = total_gain
            best_mean_deltas = mean_deltas

        if tactic == MOVE_CHOICE_TACTIC[GREEDY] and best_move is not None:
            break

    if best_move is None:
        return None, None

    return best_move, best_mean_deltas
    
def local_search(initial_solution, distance_matrix, max_iterations = None, tactic=MOVE_CHOICE_TACTIC[GREEDY]):
    distance_matrix = np.asarray(distance_matrix)
    distance_means = cost_function(distance_matrix, initial_solution)

    iteration = 0
    solution = initial_solution
    step = True

    step, deltas = local_search_move(solution, distance_means, distance_matrix, tactic)

    while step is not None and (max_iterations is None or iteration < max_iterations):
        solution = apply_move(solution, step)
        distance_means[step[1]] += deltas[0]
        distance_means[step[2]] += deltas[1]

        iteration += 1

        step, deltas = local_search_move(solution, distance_means, distance_matrix, tactic)

    return solution


def chunk(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):  
        yield arr[i:i + chunk_size]
import random
import numpy as np
from copy import deepcopy
import math 
import datetime
from functools import partial

def element_addition_cost(initial_cost, distance_matrix, solution, element, group):
    distance_gain = np.sum(distance_matrix[solution[group], element])
    return (initial_cost[0] + distance_gain), (initial_cost[1] + len(solution[group]))

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

def count_group_gain(groups, distance_matrix, element, origin, endpoint):
    distance_loss = np.sum(distance_matrix[groups[origin], element])
    distance_gain = np.sum(distance_matrix[groups[endpoint], element])

    return distance_loss - distance_gain

def get_available_moves(solution):
    moves = [
        (point, origin_idx, endpoint_idx)
        for origin_idx, origin in enumerate(solution) if len(origin) > 1
        for endpoint_idx, endpoint in enumerate(solution) if origin_idx != endpoint_idx
        for point in origin
    ]
    random.shuffle(moves)
    return moves

def get_available_candidate_moves(solution, candidateArray):
    moves = [
        (point, origin_idx, endpoint_idx)
        for origin_idx, origin in enumerate(solution) if len(origin) > 1
        for endpoint_idx, endpoint in enumerate(solution)
        if origin_idx != endpoint_idx and endpoint_idx in candidateArray[origin_idx]
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

def local_search_move(initial_solution, distance_matrix, initial_cost, tactic=MOVE_CHOICE_TACTIC[GREEDY],  cacheDict = None, candidateArray = []):
    moves = []

    best_move = None
    best_cost = None
    best_move_gain = 0
    localCacheDict = cacheDict
    after_move_cost = None

    if (len(candidateArray) > 0):
        moves = get_available_candidate_moves(initial_solution, candidateArray)
    else:
        moves = get_available_moves(initial_solution)



    for el, origin, endpoint in moves:
        total_gain = 0
        current_move = el, origin, endpoint

        if localCacheDict:
            if (el, origin, endpoint) in localCacheDict:
                if localCacheDict[(el, origin, endpoint)] < 0: continue
            else:
                localCacheDict[(el, origin, endpoint)] = count_group_gain(initial_solution, distance_matrix, el, origin, endpoint)

        after_move_cost = new_cost(initial_solution, distance_matrix, initial_cost, el, origin, endpoint)
        total_gain = initial_cost[0] / initial_cost[1] - after_move_cost[0] / after_move_cost[1]

        if total_gain > best_move_gain:
            best_move = current_move
            best_move_gain = total_gain
            best_cost = after_move_cost

        if tactic == MOVE_CHOICE_TACTIC[GREEDY] and best_move is not None:
            break

    if best_move is None:
        return None, None, localCacheDict

    return best_move, best_cost, localCacheDict

def ILS_move(initial_solution, distance_matrix, initial_cost, tactic=MOVE_CHOICE_TACTIC[GREEDY], candidateArray = []):
    pass

    
def local_search(initial_solution, distance_matrix, max_iterations = None, tactic=MOVE_CHOICE_TACTIC[GREEDY], cache = False, candidate = False):
    distance_matrix = np.asarray(distance_matrix)
    cost = cost_function(distance_matrix, initial_solution)
    cacheDict = {}
    if cache: cacheDict = {'init': 'init'}
    candidateArray = []
    randomDistanceMatrix = []

    if candidate:
        for group in initial_solution:
            elIndex = random.randint(0, len(group) - 1)
            checkFrom = group[elIndex]
            randomPointSample = random.sample(range(1, 424), 20)

            randomDistanceMatrix = [distance_matrix[checkFrom, checkTo] for checkTo in randomPointSample]
            randomDistanceMatrix, randomPointSample = zip(*sorted(zip(randomDistanceMatrix, randomPointSample)))

            randomPointSample = [x for x in randomPointSample if x not in group]
            randomPointSample = np.array(randomPointSample)[:10]

            randomGroupSample = [
                group_idx 
                for point in randomPointSample
                for group_idx, group in enumerate(initial_solution) if point in group
            ]

            randomGroupSample = list(set(randomGroupSample))
            candidateArray.append(randomGroupSample)
            # print('randomGroupSample: {}'.format(randomGroupSample))



    # print('group: {}'.format(group))
    # print('randomPointSample: {}'.format(randomPointSample))
    # print('randomGroupSample: {}'.format(randomGroupSample))
    # print('candidateArray: {}'.format(candidateArray))

    iteration = 0
    solution = initial_solution
    step = True
    if cache: 
        step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, cacheDict, candidateArray)
    else:
        step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, None, candidateArray)

    while step is not None and (max_iterations is None or iteration < max_iterations):

        if candidate:
            for group in initial_solution:
                elIndex = random.randint(0, len(group) - 1)
                checkFrom = group[elIndex]
                randomPointSample = random.sample(range(1, 424), 20)

                randomDistanceMatrix = [distance_matrix[checkFrom, checkTo] for checkTo in randomPointSample]
                randomDistanceMatrix, randomPointSample = zip(*sorted(zip(randomDistanceMatrix, randomPointSample)))

                randomPointSample = [x for x in randomPointSample if x not in group]
                randomPointSample = np.array(randomPointSample)[:5]

                randomGroupSample = [
                    group_idx 
                    for point in randomPointSample
                    for group_idx, group in enumerate(initial_solution) if point in group
                ]

                # randomGroupSample = np.array(randomGroupSample)[:5]
                randomGroupSample = list(set(randomGroupSample))
                candidateArray.append(randomGroupSample)

        solution = apply_move(solution, step)
        cost = new_cost
        iteration += 1
        if cache: 
            step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, cacheDict, candidateArray)
        else:
            step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, None, candidateArray)

    return solution, cost

def MSLS(data, random_graph_generator, n=100):
    best_cost = None
    best_solution = None
    for i in range(n):
        random_solution = random_graph_generator()
        solution, cost = local_search(random_solution, data, None, tactic=MOVE_CHOICE_TACTIC[GREEDY], cache=True, candidate=False)

        if best_cost is None or cost[0]/cost[1] < best_cost:
            best_cost = cost[0]/cost[1]
            best_solution = solution

    return best_solution, best_cost

def ILS(data, random_graph_generator, perturbation, max_time):
    best_cost = None
    best_solution = None
    start = datetime.datetime.now()
    random_solution = random_graph_generator()
    solution = random_solution

    while (datetime.datetime.now() - start).total_seconds() * 1000 < max_time:
        solution, cost = local_search(solution, data, None, tactic=MOVE_CHOICE_TACTIC[GREEDY], cache=True, candidate=False)

        if best_cost is None or cost[0]/cost[1] < best_cost:
            best_cost = cost[0]/cost[1]
            best_solution = solution

        solution = perturbation(solution)

    return best_solution, best_cost

def low_perturbation(solution, n, objects, **kargs):
    objects = np.arange(objects)
    np.random.shuffle(objects)
    to_remove = objects[:n]
    number_of_groups = len(solution)
    new_solution = [
        [ obj for obj in group if obj not in to_remove ]
        for group in solution
    ]
    new_groups = np.random.randint(0, number_of_groups, len(to_remove))
    for obj_idx, g in enumerate(new_groups):
        new_solution[g].append(to_remove[obj_idx])
    
    return new_solution

def high_perturbation(solution, n, objects, rebuild, **kargs):
    objects = np.arange(objects)
    np.random.shuffle(objects)
    to_remove = objects[:n]
    new_solution = [
        [ obj for obj in group if obj not in to_remove ]
        for group in solution
    ]
    return rebuild(new_solution, to_remove)

def NNrebuild(partial_solution, removed, data, **kwargs):
    inital_cost = cost_function(data, partial_solution)
    cost = inital_cost
    for obj in removed:
        add_to_group_cost = partial(element_addition_cost, cost, data, partial_solution, obj)
        selected_group, cost = min(
            [
                (group, add_to_group_cost(group))
                for group in range(len(partial_solution))
            ],
            key = lambda c: c[1][0] / c[1][1]
        )
        partial_solution[selected_group].append(obj)
    return partial_solution

def chunk(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):  
        yield arr[i:i + chunk_size]
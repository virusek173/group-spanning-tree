import numpy as np
import networkx as nx
import datetime
from representation import Representation
from visualization import Visualization
from controller import Controller
from local_search import local_search, cost_function, MOVE_CHOICE_TACTIC, GREEDY, STEEPEST, MSLS, ILS, low_perturbation, high_perturbation, NNrebuild
from functools import partial

def solution_to_lines(data, solution):
    lines = []
    for group in solution:
        g = nx.Graph()
        g.add_nodes_from(group)
        weights = np.asarray(data)[group,:][:,group]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                g.add_edge(group[i], group[j], weight=weights[i,j])

        lines = lines + [[[a, b] for a,b,c in nx.minimum_spanning_edges(g)]]

    return lines

def multiLaunch(rep, initial_solution_fn, tactic, name, n, cache = None, candidate=False):
    matrixData = rep.getMatrixData()


    n_sum = 0
    n_max = float("-inf")
    n_min = float("inf")
    t_sum = 0
    t_max = float("-inf")
    t_min = float("inf")
    max_solution = None
    min_solution = None

    for i in range(n):
        print(name, ' Solution ', i+1)
        plotData = initial_solution_fn(matrixData)
        initial_cost = cost_function(np.asarray(matrixData), plotData)
        start = datetime.datetime.now()
        final_solution = local_search(plotData, matrixData, None, tactic, cache, candidate)
        final_time = (datetime.datetime.now() - start).total_seconds() * 1000
        final_cost = cost_function(np.asarray(matrixData), final_solution)
        final_cost = final_cost[0] / final_cost[1]

        n_sum += final_cost
        if final_cost < n_min:
            n_min = final_cost
            min_solution = final_solution

        if final_cost > n_max:
            n_max = final_cost
            max_solution = final_solution

        t_sum += final_time
        if final_time < t_min:
            t_min = final_time
        if final_time > t_max:
            t_max = final_time
               
        print('Final solution cost:', final_cost, 'Time: ', final_time)

    print('Cost min: {}, max: {}, avg: {}'.format(n_min, n_max, n_sum / n))
    print('Time min: {}, max: {}, avg: {}'.format(t_min, t_max, t_sum / n))

    coordData = rep.getCoordData()
    visual = Visualization(coordData, solution_to_lines(matrixData, min_solution))
    visual.showScatterplotFromDict(name)

def singleLaunch(rep, initial_solution_fn, tactic, name, cache = False, candidate = False):
    matrixData = rep.getMatrixData()
    plotData = initial_solution_fn(matrixData)
    start = datetime.datetime.now()

    final_solution = local_search(plotData, matrixData, None, tactic, cache, candidate)
    final_time = (datetime.datetime.now() - start).total_seconds() * 1000
    final_cost = cost_function(np.asarray(matrixData), final_solution)
    final_cost = final_cost[0] / final_cost[1]

    print('Final solution cost:', final_cost, 'Time: ', final_time)

    coordData = rep.getCoordData()
    visual = Visualization(coordData, solution_to_lines(matrixData, final_solution))
    visual.showScatterplotFromDict(name)



# multiLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[GREEDY], 'output/nn_greedy', 5)
# multiLaunch(rep, controller.createGraphsRandomMethod, MOVE_CHOICE_TACTIC[GREEDY], 'output/random_greedy', 10)
# multiLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/nn_steepest', 5)
#multiLaunch(rep, controller.createGraphsRandomMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/random_steepest', 25)

# multiLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/nn_WithoutModifications', 10)
# singleLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], None, 'output/WithoutModifications')

# multiLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/nn_WithCache', 10, True)
# singleLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/WithCache', True)

# multiLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/nn_WithCandidate', 5, False, True)
# singleLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/WithCandidate', False, True)

# multiLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/nn_WithCacheAndCandidate', 10,  True, True)
# singleLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/WithCacheAndCandidate', True, True)

def run_MSLS(save_file='result', n = 1):
    rep = Representation()
    data = rep.getMatrixData()
    controller = Controller(20, len(data))

    solutions = []
    costs = []
    durations = [] 

    for i in range(n):
        start = datetime.datetime.now()
        print(start)
        solution, cost = MSLS(data, controller.createGraphsRandomMethod, n=100)
        duration = (datetime.datetime.now() - start).total_seconds() * 1000
        durations.append(duration)
        solutions.append(solution)
        costs.append(cost)


    print('Koszt & Czas')
    print('min & max & średnia & min & max & średnia')
    print('{} & {} & {} & {} & {} & {}'.format(
        np.min(np.array(costs)).round(2),
        np.max(np.array(costs)).round(2),
        np.mean(np.array(costs)).round(2),
        np.min(np.array(durations)).round(0),
        np.max(np.array(durations)).round(0),
        np.mean(np.array(durations)).round(0)
    ))
    best_solution = solutions[np.where(np.asarray(costs) == np.min(costs))[0][0]]
    coordData = rep.getCoordData()
    visual = Visualization(coordData, solution_to_lines(data, best_solution))
    visual.showScatterplotFromDict(save_file)

def run_ILS_low(n = 1, max_time=1000, save_file='resultILS_low'):
    rep = Representation()
    data = rep.getMatrixData()
    controller = Controller(20, len(data))

    solutions = []
    costs = []
    durations = [] 
    perturbation = partial(low_perturbation, n=8, objects=424)

    for i in range(n):
        start = datetime.datetime.now()
        print(start)
        solution, cost = ILS(data, controller.createGraphsRandomMethod, perturbation, max_time=max_time)
        duration = (datetime.datetime.now() - start).total_seconds() * 1000
        durations.append(duration)
        solutions.append(solution)
        costs.append(cost)


    print('Koszt & Czas')
    print('min & max & średnia & min & max & średnia')
    print('{} & {} & {} & {} & {} & {}'.format(
        np.min(np.array(costs)).round(2),
        np.max(np.array(costs)).round(2),
        np.mean(np.array(costs)).round(2),
        np.min(np.array(durations)).round(0),
        np.max(np.array(durations)).round(0),
        np.mean(np.array(durations)).round(0)
    ))
    best_solution = solutions[np.where(np.asarray(costs) == np.min(costs))[0][0]]
    coordData = rep.getCoordData()
    visual = Visualization(coordData, solution_to_lines(data, best_solution))
    visual.showScatterplotFromDict(save_file)

def run_ILS_high(n = 1, max_time = 1000, save_file='resultILS_high'):
    rep = Representation()
    data = rep.getMatrixData()
    controller = Controller(20, len(data))

    solutions = []
    costs = []
    durations = [] 
    rebuild = partial(NNrebuild, data=np.asarray(data))
    perturbation = partial(high_perturbation, n=84, objects=424, rebuild=rebuild)

    for i in range(n):
        start = datetime.datetime.now()
        print(start)
        solution, cost = ILS(data, controller.createGraphsRandomMethod, perturbation, max_time=max_time)
        duration = (datetime.datetime.now() - start).total_seconds() * 1000
        durations.append(duration)
        solutions.append(solution)
        costs.append(cost)


    print('Koszt & Czas')
    print('min & max & średnia & min & max & średnia')
    print('{} & {} & {} & {} & {} & {}'.format(
        np.min(np.array(costs)).round(2),
        np.max(np.array(costs)).round(2),
        np.mean(np.array(costs)).round(2),
        np.min(np.array(durations)).round(0),
        np.max(np.array(durations)).round(0),
        np.mean(np.array(durations)).round(0)
    ))
    best_solution = solutions[np.where(np.asarray(costs) == np.min(costs))[0][0]]
    coordData = rep.getCoordData()
    visual = Visualization(coordData, solution_to_lines(data, best_solution))
    visual.showScatterplotFromDict(save_file)

# run_MSLS(n=10, save_file='result')
run_ILS_low(n=10, max_time=707936)
run_ILS_high(n=10, max_time=707936)
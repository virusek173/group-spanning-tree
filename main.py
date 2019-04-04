import numpy as np
import networkx as nx
import datetime
from representation import Representation
from visualization import Visualization
from controller import Controller
from local_search import local_search, cost_function, MOVE_CHOICE_TACTIC, GREEDY, STEEPEST

def singleLaunch(rep, initial_solution_fn, tactic, name, n):
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
        final_solution = local_search(plotData, matrixData, None, tactic)
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


    lines = []
    for group in min_solution:
        g = nx.Graph()
        g.add_nodes_from(group)
        weights = np.asarray(matrixData)[group,:][:,group]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                g.add_edge(group[i], group[j], weight=weights[i,j])

        lines = lines + [[[a, b] for a,b,c in nx.minimum_spanning_edges(g)]]

    coordData = rep.getCoordData()
    visual = Visualization(coordData, lines)
    visual.showScatterplotFromDict(name)

rep = Representation()
matrixData = rep.getMatrixData()
controller = Controller(20, len(matrixData))

# singleLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[GREEDY], 'output/nn_greedy', 50)
# singleLaunch(rep, controller.createGraphsRandomMethod, MOVE_CHOICE_TACTIC[GREEDY], 'output/random_greedy', 10)
singleLaunch(rep, controller.createGraphsNearestNeighborMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/nn_steepest', 20)
#singleLaunch(rep, controller.createGraphsRandomMethod, MOVE_CHOICE_TACTIC[STEEPEST], 'output/random_steepest', 25)
import numpy as np

from representation import Representation
from visualization import Visualization
from controller import Controller
from local_search import local_search, cost_function

def singleLaunch():
    rep = Representation()
    matrixData = rep.getMatrixData()
    controller = Controller(20, len(matrixData))

    # plotData = controller.createGraphsNearestNeighborMethod(matrixData)
    plotData = controller.createGraphsRandomMethod()
    
    print('Problem data', len(plotData), len(matrixData))
    print('Initial solution cost:', np.sum(cost_function(np.asarray(matrixData), plotData)))
    final_solution = local_search(plotData, matrixData)
    print('Final solution cost:', np.sum(cost_function(np.asarray(matrixData), final_solution)))
    print([len(group) for group in final_solution])

def hundredTest():
    rep = Representation()
    matrixData = rep.getMatrixData()
    controller = Controller(20, len(matrixData))
    
    # plotData = controller.testRegretMethod(matrixData)
    plotData = controller.testMethodNeighbor(matrixData)

    coordData = rep.getCoordData()
    visual = Visualization(coordData, plotData)
    visual.showScatterplotFromDict()

singleLaunch()
# hundredTest()

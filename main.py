from representation import Representation
from visualization import Visualization
from controller import Controller

def singleLaunch():
    rep = Representation()
    controller = Controller()
    matrixData = rep.getMatrixData()
    
    plotData = controller.createGraphsRegretMethod(matrixData)
    # plotData = controller.createGraphsNearestNeighborMethod(matrixData)

    distance = controller.countDistance(plotData, matrixData)
    print('sum of distances: {}'.format(distance))

    coordData = rep.getCoordData()
    visual = Visualization(coordData, plotData)
    visual.showScatterplotFromDict()

def hundredTest():
    rep = Representation()
    controller = Controller()
    matrixData = rep.getMatrixData()
    
    plotData = controller.testRegretMethod(matrixData)
    # plotData = controller.testMethodNeighbor(matrixData)

    coordData = rep.getCoordData()
    visual = Visualization(coordData, plotData)
    visual.showScatterplotFromDict()

singleLaunch()
# hundredTest()

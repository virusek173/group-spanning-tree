from representation import Representation
from visualization import Visualization
from controller import Controller

def launch():
    rep = Representation()
    controller = Controller()
    matrixData = rep.getMatrixData()
    
    # plotData = controller.test()

    plotData = controller.createGraphsRegretMethod(matrixData)
    # distance = controller.countDistance(plotData, matrixData)
    # print('sum of distances: {}'.format(distance))
    
    # plotData = controller.testMethod(matrixData)
    # print('sum of distances: {}'.format(distance))


    coordData = rep.getCoordData()
    visual = Visualization(coordData, plotData)
    visual.showScatterplotFromDict()

launch()
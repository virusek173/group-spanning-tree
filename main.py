from representation import Representation
from visualization import Visualization

def launch():
    rep = Representation()
    plotData = rep.createGraphsRandomMethod()

    coordData = rep.getCoordData()
    visual = Visualization(coordData, plotData)
    visual.showScatterplot()

launch()
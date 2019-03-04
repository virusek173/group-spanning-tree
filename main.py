from representation import Representation
from visualization import Visualization

def launch():
    fileName = 'objects.data'

    rep = Representation(fileName)
    coordData = rep.getCoordData()

    visual = Visualization(coordData)
    visual.showScatterplot()

launch()
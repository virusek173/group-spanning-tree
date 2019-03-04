import random
import numpy as np
import matplotlib.pyplot as plt

class Visualization:
    coordData=()
    plotData=[]

    def __init__(self, coordData, plotData):
        self.coordData = coordData
        self.plotData = plotData

    def showScatterplot(self):
        coordData = self.coordData
        rootPoints = random.sample(range(1, 201), 200)

        colors = (0,0,0)
        pointSize = np.pi*3

        # Visualize ponts
        scatterX = [x[0] for x in coordData]
        scatterY = [y[1] for y in coordData]

        plt.scatter(scatterX, scatterY, s=pointSize, c=colors, alpha=0.5)
        
        # Visualize graphs
        for graph in self.plotData:
            plotX = [coordData[index][0] for index in graph]
            plotY = [coordData[index][1] for index in graph]
            plt.plot(plotX, plotY)

        # colors = (1,0,0)
        # pointSize = np.pi*5

        # Visualize root points
        # scatterX = [coordData[index][0] for index in rootPoints]
        # scatterY = [coordData[index][1] for index in rootPoints]
        # plt.scatter(scatterX, scatterY, s=pointSize, c=colors, alpha=0.8)

        plt.title('Spanning tree groups')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.savefig('output/random_points.png')
        print('showing visualization...')
        plt.show()
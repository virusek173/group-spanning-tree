import random
import numpy as np
import matplotlib.pyplot as plt

import math
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, Oval
from bokeh.palettes import Spectral8

class Visualization:
    coordData=()
    plotData=[]

    def __init__(self, coordData, plotData):
        self.coordData = coordData
        self.plotData = plotData

    def showScatterplot(self):
        coordData = self.coordData
        rootPoints = [item[0] for item in self.plotData]

        # random.sample(range(1, 201), 20)

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

        colors = (1,0,0)
        pointSize = np.pi*4

        # Visualize root points
        scatterX = [coordData[index][0] for index in rootPoints]
        scatterY = [coordData[index][1] for index in rootPoints]
        plt.scatter(scatterX, scatterY, s=pointSize, c=colors, alpha=0.8)

        plt.title('Spanning tree groups')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/regret100xtests_points.png')
        print('showing visualization...')
        plt.show()

    def showScatterplotFromDict(self):
        coordData = self.coordData
        data = [[12,16], [12,18], [30,18], [10,20]]
        # rootPoints = [item[0] for item in self.plotData]

        # random.sample(range(1, 201), 20)

        colors = (0,0,0)
        pointSize = np.pi*6

        # Visualize points
        scatterX = [x[0] for x in coordData]
        scatterY = [y[1] for y in coordData]

        plt.scatter(scatterX, scatterY, s=pointSize, c=colors, alpha=0.5)
        # print(self.plotData)
        # Visualize graphs
        for line in self.plotData:
            plotX = [coordData[index][0] for index in line]
            plotY = [coordData[index][1] for index in line]
            plt.plot(plotX, plotY, color='blue')

        colors = (1,0,0)
        pointSize = np.pi*4

        plt.title('Spanning tree groups')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/random_REAL_100_test_MST.png')
        print('showing visualization...')
        plt.show()
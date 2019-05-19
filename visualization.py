import random
import numpy as np
import matplotlib.pyplot as plt

import math

pallete = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
]
class Visualization:
    coordData=()
    plotData=[]

    def __init__(self, coordData, plotData):
        self.coordData = coordData
        self.plotData = plotData

    def showSimilarityPlot(self, similarityArray, savename):
        simiarityData = [item.similarity for item in  similarityArray]
        costData = [item.cost for item in  similarityArray]

        plt.clf()
        plt.title('Similarity and result plot')
        plt.plot(costData, simiarityData)
        plt.xlabel('Result')
        plt.ylabel('Similarity')
        plt.savefig(savename)


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

    def showScatterplotFromDict(self, savename='output/random_REAL_100_test_MST.png'):
        coordData = self.coordData
        data = [[12,16], [12,18], [30,18], [10,20]]
        # rootPoints = [item[0] for item in self.plotData]

        # random.sample(range(1, 201), 20)

        colors = (0,0,0)
        pointSize = np.pi*6

        # Visualize points
     

        # print(self.plotData)
        # Visualize graphs
        for idx, group in enumerate(self.plotData):
            for line in group:
                plotX = [coordData[index][0] for index in line]
                plotY = [coordData[index][1] for index in line]
                plt.plot(plotX, plotY, color=pallete[idx])

          
        scatterX = [x[0] for x in coordData]
        scatterY = [y[1] for y in coordData]
        plt.scatter(scatterX, scatterY, s=pointSize, c=pallete[idx], alpha=0.5)
        colors = (1,0,0)
        pointSize = np.pi*4

        plt.title('Groups')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(savename)

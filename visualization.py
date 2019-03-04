import numpy as np
import matplotlib.pyplot as plt

class Visualization:
    coordData=()

    def __init__(self, coordData):
        self.coordData = coordData

    def showScatterplot(self):
        coordData = self.coordData
        
        x = [x[0] for x in coordData]
        y = [x[1] for x in coordData]
        colors = (0,0,0)
        pointSize = np.pi*3

        plt.scatter(x, y, s=pointSize, c=colors, alpha=0.5)
        plt.title('Initial scatter plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/init_scatterpoints_plot.png')
        print('showing visualization...')
        plt.show()
import os
import random
from scipy.spatial import distance as countDistance

class Representation:
    fileCoordsName = 'objects20_06.data'
    fileMatrixName = 'matrix20_06.data'
    coordData = ()
    matrixData = ()

    def __init__(self):
        # Tests:
        # self.__testCountDistance()

        print('Reading coords from file...')
        self.coordData = self.__readFileToCoords()

        if self.__checkMatrixFileExist():
            print('Reading matrix from file...')
            self.matrixData = self.__readFileToMatrix()
            print('Done! {} long matrix loaded'.format(len(self.matrixData)))
        else:
            print('Creating matrix from file...')
            self.matrixData = self.__createNeighborhoodMatrix()
            print('Saving matrix to file...')
            self.__saveMatrixToFile(self.matrixData)
            print('Saved! {} long matrix'.format(len(self.matrixData)))


    def __checkMatrixFileExist(self):
        if (os.path.isfile('./data/{}'.format(self.fileMatrixName))):
            return True

        return False

    def __readFileToCoords(self):
        finalData = ()
        with open('./data/{}'.format(self.fileCoordsName), 'r') as f:
            data = f.readlines()

            for line in data:
                coordinates = tuple(line.split())
                x = int(coordinates[0])
                y = int(coordinates[1])
                finalData = finalData + ((x, y), )

        return finalData

    def __readFileToMatrix(self):
        finalData = []
        with open('./data/{}'.format(self.fileMatrixName), 'r') as f:
            data = f.readlines()

            for line in data:
                # lineValues = line.split()
                lineValues = [float(x) for x in line.split()]
                finalData.append(lineValues)

        return finalData


    def __createNeighborhoodMatrix(self):
        finalData = []
        coordData = self.coordData

        finalData = [
            [self.__countDistance(currentItem, item) for item in coordData] 
            for currentItem in coordData]
        

        return finalData

    def __countDistance(self, x, y):
        distance = 0
        distance = countDistance.euclidean(x, y)

        return distance

    def __saveMatrixToFile(self, matrixData):
        with open('./data/{}'.format(self.fileMatrixName), 'a+') as f:
            for line in matrixData:
                for distance in line:
                    f.write('{} '.format(distance))
                f.write('\n')

    def __testCountDistance(self):
        x = (1,13)
        y = (5,13)
        expectedValue = 4
        value = self.__countDistance(x,y)

        if value == expectedValue:
            print('YES')
        else:
            print('NO! {}'.format(value))

    def divideChunks(self, arr, chunk): 
        for i in range(0, len(arr), chunk):  
            yield arr[i:i + chunk]

        

    def createGraphsRandomMethod(self):
        allRandomPoints = random.sample(range(1, 201), 200)
        plotData = list(self.divideChunks(allRandomPoints, 20))
        # check if sum is equal points number

        return plotData

    def getMatrixData(self):
        return self.matrixData

    def getCoordData(self):
        return self.coordData
import os
import random
import numpy as np
from copy import deepcopy

# from scipy.spatial import distance as countDistance

class Controller:
    def _divideChunks(self, arr, chunk): 
        for i in range(0, len(arr), chunk):  
            yield arr[i:i + chunk]

    def createGraphsRandomMethod(self):
        allRandomPoints = random.sample(range(1, 201), 200)
        plotData = list(self._divideChunks(allRandomPoints, 20))
        # check if sum is equal points number

        return plotData

    def _minValueNotZero(self, myList):
        minVal = 99999
        # print(myList)
        for x in myList:
            if float(x) != float(0) and x < minVal:
                minVal = x

        # print('choosed')
        # print(minVal)

        # if float(minVal) == float(0):
        #     print('end with IS YES AS 0')
        #     print(minVal)
        # else:
        #     print('end with NOT 0')
        #     print(minVal)
        #     print('myList')
        #     print(myList)
        #     print('INDEX IN FUNCTION')
        #     print(myList.index(minVal))

        # print('minVal')
        # print(minVal)

        return minVal

    def test(self):
        tab = [215.87033144922904, 44.9221548904324, 34.713109915419565, 0.0, 48.25971404805462]
        tab2 = [215.87033144922904, 12.9221548904324, 34.713109915419565, 0.0, 48.25971404805462]
        # tab = [19.924858845171276, 12.083045973594572]
        # countRegret = self._countRegret(tab, 18)
        
        minIndex = tab.index(self._minValueNotZero(tab))
        minIndex2 = tab2.index(self._minValueNotZero(tab2))


        print(minIndex)
        print(minIndex2)
        print('------>Test ended <--------')


    def _zeroColumn(self, matrixData, indexToDel):
        localMatrixData = list(matrixData)

        for colIndex, col in enumerate(localMatrixData):
            if colIndex == indexToDel:
                for rowIndex, rowEl in enumerate(col):
                    localMatrixData[rowIndex][colIndex] = 0.0

        return localMatrixData
        

    def _countRegret(self, sortedArray, number):
        regret = 0
        numberToCheck = 20 - number -1

        for index, element in enumerate(sortedArray):
            if index == 0: continue
            if numberToCheck == 0: break

            if float(element) == float(0): continue
            regret += abs(sortedArray[index-1] - element)
            numberToCheck -= 1

        return regret


    def createGraphsRegretMethod(self, matrixData):
        localMatrixData = deepcopy(matrixData)
        allRandomPoints = random.sample(range(1, 201), 10)
        # [[a],[b],...,[z]] 10 groups of 1 el.
        plotData = list(self._divideChunks(allRandomPoints, 1))
        for z in plotData:
            localMatrixData = self._zeroColumn(localMatrixData, z[0])


        for index, itemList in enumerate(plotData):
            minIndex = 0
            for item in itemList:
                row = localMatrixData[item]
                minIndex = row.index(self._minValueNotZero(row))
            
            plotData[index].append((minIndex))
            localMatrixData = self._zeroColumn(localMatrixData, minIndex)


        # for x in range(0, 20):
        needBreak = False
        steps = 0
        while True: 
            steps +=2
            if needBreak: break

            for index, itemList in enumerate(plotData):
                maxRegretIndex = 0
                maxRegret = 0
                minArray = []
                regretArray = []
                test = np.array(localMatrixData[0])
                if np.all(test==0): 
                    needBreak = True
                    break

                for itemIndex, item in enumerate(itemList):
                    row = localMatrixData[item]
                    minIndex = row.index(self._minValueNotZero(row))
                    minArray.append(minIndex)

                    rowCopy = row.copy()
                    rowCopy.sort()

                    regret = self._countRegret(rowCopy, len(itemList))

                    if maxRegret < regret:
                        maxRegret = regret
                        maxRegretIndex = itemIndex

                distance = localMatrixData[itemList[maxRegretIndex]][minArray[maxRegretIndex]]
                if distance < steps:
                    plotData[index].insert(maxRegretIndex, minArray[maxRegretIndex])
                    localMatrixData = self._zeroColumn(localMatrixData, minArray[maxRegretIndex])
        return plotData


    def createGraphsNearestNeighborMethod(self, matrixData):
        localMatrixData = deepcopy(matrixData)
        allRandomPoints = random.sample(range(1, 201), 10)
        # [[a],[b],...,[z]] 10 groups of 1 el.
        plotData = list(self._divideChunks(allRandomPoints, 1))
        for z in plotData:
            localMatrixData = self._zeroColumn(localMatrixData, z[0])

        needBreak = False
        steps = 0
        while True: 
            steps +=2
            if needBreak: break
            for index, itemList in enumerate(plotData):
                minIndex = 0

                test = np.array(localMatrixData[0])
                if np.all(test==0): 
                    needBreak = True
                    break

                for item in itemList:
                    row = localMatrixData[item]
                    minIndex = row.index(self._minValueNotZero(row))
                
                plotData[index].append((minIndex))
                localMatrixData = self._zeroColumn(localMatrixData, minIndex)
                
        return plotData


    def countDistance(self, plotData, matrixData):
        sumOfDistances = 0

        for itemList in plotData:
            for index, item in enumerate(itemList):
                if index == 0: continue
                sumOfDistances += matrixData[itemList[index-1]][item]
        return sumOfDistances


    def testMethod(self, matrixData):
        minDistance = 10000
        finalPlotData = []

        for x in range(0, 100):
            plotData = self.createGraphsRegretMethod(matrixData)
            distance = self.countDistance(plotData, matrixData)
            if distance < minDistance:
                minDistance = distance
                finalPlotData = plotData

            print('iteration {} Current distance: {}'.format(x, distance))
            print('iteration {} Minimum distance: {}'.format(x, minDistance))
            print('>>>>>>>---------<<<<<<<<')

        
        print('Final distance: '.format(minDistance))

        return finalPlotData
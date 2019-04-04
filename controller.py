import os
import random
import time
import numpy as np
from copy import deepcopy
import math
# from scipy.spatial import distance as countDistance

class Controller:
    def __init__(self, groups, n):
        self.groups = groups
        self.n = n

    def _divideChunks(self, arr, chunk): 
        for i in range(0, len(arr), chunk):  
            yield arr[i:i + chunk]

    def createGraphsRandomMethod(self, *args):
        allRandomPoints = random.sample(range(0, self.n), self.n)
        plotData = list(self._divideChunks(allRandomPoints, math.ceil(self.n / self.groups)))
        return plotData

    def _minValueNotZero(self, myList):
        minVal = 9999

        for x in myList:
            if float(x) != float(0) and x < minVal:
                minVal = x

        return minVal

    def test(self):
        tab = [215.87033144922904, 44.9221548904324, 34.713109915419565, 0.0, 48.25971404805462]
        tab2 = [215.87033144922904, 12.9221548904324, 34.713109915419565, 0.0, 48.25971404805462]
        
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
        numberToCheck = 20 - number - 1

        for index, element in enumerate(sortedArray):
            if index == 0: continue
            if numberToCheck == 0: break

            if float(element) == float(0): continue
            regret += abs(sortedArray[index-1] - element)
            numberToCheck -= 1

        return regret


    def createGraphsRegretMethod(self, matrixData):
        localMatrixData = deepcopy(matrixData)
        allRandomPoints = random.sample(range(0, self.n), self.groups)
        # [[a],[b],...,[z]] 10 groups of 1 el.
        plotData = list(self._divideChunks(allRandomPoints, 1))
        plotToVisualize = []

        for z in plotData:
            localMatrixData = self._zeroColumn(localMatrixData, z[0])


        for index, itemList in enumerate(plotData):
            minIndex = 0
            for item in itemList:
                row = localMatrixData[item]
                minValue = self._minValueNotZero(row)
                minIndex = row.index(minValue)
            
            plotData[index].append((minIndex))
            plotToVisualize.append([plotData[index][0], minIndex])
            # print(plotToVisualize)
            localMatrixData = self._zeroColumn(localMatrixData, minIndex)


        needBreak = False
        steps = 0
        # for x in range(0, 4):
        while True: 
            steps += 1
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

                    minValue = self._minValueNotZero(row)
                    minIndex = row.index(minValue)

                    minArray.append(minIndex)

                    rowCopy = row.copy()
                    rowCopy.sort()

                    regret = self._countRegret(rowCopy, len(itemList))

                    if maxRegret > regret:
                        maxRegret = regret
                        maxRegretIndex = itemIndex


                distance = localMatrixData[itemList[maxRegretIndex]][minArray[maxRegretIndex]]
                
                if distance < steps:
                    plotToVisualize.append([minArray[maxRegretIndex], itemList[maxRegretIndex]])
                    plotData[index].insert(maxRegretIndex, minArray[maxRegretIndex])
                    localMatrixData = self._zeroColumn(localMatrixData, minArray[maxRegretIndex])
                    
        return plotToVisualize


    def createGraphsNearestNeighborMethod(self, matrixData):
        localMatrixData = deepcopy(matrixData)
        allRandomPoints = random.sample(range(0, self.n), self.groups)
        # [[a],[b],...,[z]] 10 groups of 1 el.
        plotData = list(self._divideChunks(allRandomPoints, 1))
        plotToVisualize = []

        for z in plotData:
            localMatrixData = self._zeroColumn(localMatrixData, z[0])

        for index, itemList in enumerate(plotData):
            minIndex = 0
            for item in itemList:
                row = localMatrixData[item]
                minValue = self._minValueNotZero(row)
                minIndex = row.index(minValue)
            
            plotData[index].append((minIndex))
            plotToVisualize.append([plotData[index][0], minIndex])
            localMatrixData = self._zeroColumn(localMatrixData, minIndex)


        needBreak = False
        steps = 0
        while True: 
            steps += 1
            if needBreak: break

            for index, itemList in enumerate(plotData):
                maxRegretIndex = 0
                maxRegret = 0
                minValue = 1000
                fromIndex = 0
                minIndex = 0

                regretArray = []
                test = np.array(localMatrixData[0])
                if np.all(test == 0): 
                    needBreak = True
                    break

                for itemIndex, item in enumerate(itemList):
                    row = localMatrixData[item]

                    tempMinValue = self._minValueNotZero(row)
                    if minValue > tempMinValue:
                        minValue = tempMinValue
                        minIndex = row.index(minValue)
                        fromIndex = item

                distance = localMatrixData[itemList[itemIndex]][minIndex]
                
                if distance < steps:
                    plotToVisualize.append([itemList[itemIndex], minIndex])
                    plotData[index].insert(fromIndex, minIndex)
                    localMatrixData = self._zeroColumn(localMatrixData, minIndex)
                    
        return plotData


    def countDistance(self, plotData, matrixData):
        sumOfDistances = 0

        for itemList in plotData:
            sumOfDistances += matrixData[itemList[0]][itemList[1]]
        return sumOfDistances


    def testRegretMethod(self, matrixData):
        minDistance = 10000
        maxDistance = 0
        avgDistance = 0
        sumDistance = 0

        minTime = 10000
        maxTime = 0
        avgTime = 0
        sumTime = 0

        finalPlotData = []

        for x in range(0, 100):
            start_time = time.time()
            plotData = self.createGraphsRegretMethod(matrixData)
            elapsed_time = time.time() - start_time

            milli_sec = int(round(elapsed_time * 1000))
            distance = self.countDistance(plotData, matrixData)
            sumDistance += distance
            sumTime += milli_sec

            if distance < minDistance:
                minDistance = distance
                finalPlotData = plotData

            if distance > maxDistance:
                maxDistance = distance

            if milli_sec < minTime:
                minTime = milli_sec

            if milli_sec > maxTime:
                maxTime = milli_sec

            print('iteration {} Current distance: {}'.format(x, distance))
            print('iteration {} Minimum distance: {}'.format(x, minDistance))
            print('iteration {} MIN Time of execution: {}'.format(x, minTime))
            print('iteration {} MAX Time of execution: {}'.format(x, maxTime))
            print('iteration {} Time of execution: {}'.format(x, milli_sec))
            print('>>>>>>>---------<<<<<<<<')

        avgDistance = sumDistance/100
        avgTime = sumTime/100

        print('Final minDistance: {}'.format(minDistance))
        print('Final maxDistance: {}'.format(maxDistance))
        print('Final avgDistance: {}'.format(avgDistance))

        print('Final minTime: {}'.format(minTime))
        print('Final maxTime: {}'.format(maxTime))
        print('Final avgTime: {}'.format(avgTime))

        return finalPlotData

    def testMethodNeighbor(self, matrixData):
        minDistance = 10000
        maxDistance = 0
        avgDistance = 0
        sumDistance = 0

        minTime = 10000
        maxTime = 0
        avgTime = 0
        sumTime = 0

        finalPlotData = []

        for x in range(0, 100):
            start_time = time.time()
            plotData = self.createGraphsNearestNeighborMethod(matrixData)
            elapsed_time = time.time() - start_time

            milli_sec = int(round(elapsed_time * 1000))
            distance = self.countDistance(plotData, matrixData)
            sumDistance += distance
            sumTime += milli_sec

            if distance < minDistance:
                minDistance = distance
                finalPlotData = plotData

            if distance > maxDistance:
                maxDistance = distance

            if milli_sec < minTime:
                minTime = milli_sec

            if milli_sec > maxTime:
                maxTime = milli_sec

            print('iteration {} Current distance: {}'.format(x, distance))
            print('iteration {} Minimum distance: {}'.format(x, minDistance))
            print('iteration {} MIN Time of execution: {}'.format(x, minTime))
            print('iteration {} MAX Time of execution: {}'.format(x, maxTime))
            print('iteration {} Time of execution: {}'.format(x, milli_sec))
            print('>>>>>>>---------<<<<<<<<')

        avgDistance = sumDistance/100
        avgTime = sumTime/100

        print('Final minDistance: {}'.format(minDistance))
        print('Final maxDistance: {}'.format(maxDistance))
        print('Final avgDistance: {}'.format(avgDistance))

        print('Final minTime: {}'.format(minTime))
        print('Final maxTime: {}'.format(maxTime))
        print('Final avgTime: {}'.format(avgTime))

        return finalPlotData
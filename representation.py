class Representation:
    coordData=()
    fileName=''

    def __init__(self, fileName):
        self.fileName = fileName
        self.coordData = self.__readFromFile(fileName)

    def __readFromFile(self, fileName):
        finalData=()
        with open('./data/{}'.format(fileName), 'r') as f:
            data = f.readlines()
        
            for line in data:
                coordinates = tuple(line.split())
                x = int(coordinates[0])
                y = int(coordinates[1])
                finalData = finalData + ((x, y), )

        return finalData


    def getCoordData(self):
        return self.coordData
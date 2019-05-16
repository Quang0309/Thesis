
from enum import Enum
import numpy as np


class Dictionary():
    def __init__(self):
        self.mydict = {}
        self.value = 1
        with open("dependency.txt", "a+" ,encoding="utf8") as myFile:
            myFile.seek(0)
            lines = myFile.read()
            for line in lines.split():
                if (line not in ['\n', '\r\n']):
                    line = line.replace("\n","")
                    print(line)
                    self.mydict[line] = self.value
                    self.value += 1
            print(self.mydict)    

        self.unvisited = []

    def getDependencyValue(self, dependency):
        value = self.mydict.get(dependency)
        if (value == None):
            self.mydict[dependency] = self.value
            value = self.value
            self.value += 1
            with open("dependency.txt", "a+" ,encoding="utf8") as myFile:
                dependency = dependency + '\n'
                myFile.write(dependency)

            return value
        else:
            return value


class AdjGenerator:

    def __init__(self, fileName, numberOfWords=50, batchSize=80):
        self.fileName = fileName
        self.dictionary = Dictionary()
        self.numberOfWords = numberOfWords
        self.batchSize = batchSize
        self.inputFile = open(self.fileName, "r", encoding="utf8")

    def generateMatrix(self):
        arrayList = []
        arrayOfASentence = []

        for line in self.inputFile:
            if (line in ['\n', '\r\n']):
                print("Empty line! End of a sentence.")
                if (arrayOfASentence != []):
                    arrayList.append(arrayOfASentence)
                arrayOfASentence = []
                if len(arrayList) == self.batchSize:
                    print("Completed a batch !!!")
                    break
            else:  # remove the line delimiter "\n" at the end of this line
                line = line.replace("\n", "")
            print(line)

            start = 0
            end = 0
            isFirstNumber = True
            firstNumber = -1
            secondNumber = -1
            dependency = ''

            for index, character in enumerate(line):
                # print(character)
                if (character == '('):  # Dependency
                    if (line[index + 1] != '('):  # check the char next to the '('
                        dependency = line[0:index]
                        print("Dependency:")
                        print(dependency)
                    else:
                        break

                elif (character == '-'):
                    start = index + 1
                elif ((character == ',' and line[index + 1] == ' ') or character == ')'):
                    end = index
                    if (isFirstNumber):
                        i = 1
                        while (True):
                            if (line[index - i] == "'"):
                                end = end - 1
                                i += 1
                            else:
                                break
                        
                        firstNumber = int(line[start:end])
                        print("Number: ")
                        print(firstNumber)
                        isFirstNumber = False
                    else:
                        i = 1
                        while (True):
                            if (line[index - i] == "'"):
                                end = end - 1
                                i += 1
                            else:
                                break
                                                 
                        secondNumber = int(line[start:end])
                        print("Second number:")
                        print(secondNumber)

                        if (dependency != 'root'):
                            # Creating an array with form (dependencyNumber, firstNumber, secondNumber) for this dependency
                            dependencyArray = [self.dictionary.getDependencyValue(dependency),
                                               firstNumber - 1,
                                               secondNumber - 1]
                            print("array: ")
                            print(dependencyArray)
                            arrayOfASentence.append(dependencyArray)                    

        # print("Array list: ")
        # if (arrayOfASentence != []):
        #     arrayList.append(arrayOfASentence)
        # print(arrayList)

        # Shift the index of the word
        for index, arrayOfASentence in enumerate(arrayList):
            shiftValue = index * self.numberOfWords
            for dependencyArray in arrayOfASentence:
                dependencyArray[1] += shiftValue
                dependencyArray[2] += shiftValue

        print("Array list after shifting value: ")
        print(arrayList)

        # Create the matrix
        numberOfSentence = len(arrayList)
        print("Number of sentences:")
        print(numberOfSentence)
        numberOfRows = self.numberOfWords * self.batchSize
        print("Number of rows:")
        print(numberOfRows)
        labelMatrix = np.zeros((numberOfRows, numberOfRows))
        adjMatrix = np.zeros((numberOfRows, numberOfRows))
        labelInverseMatrix = np.zeros((numberOfRows, numberOfRows))
        adjInverseMatrix = np.zeros((numberOfRows, numberOfRows))

        # Fill value to this matrix
        for index, arrayOfASentence in enumerate(arrayList):
            for dependencyArray in arrayOfASentence:
                labelMatrix[dependencyArray[1]
                            ][dependencyArray[2]] = dependencyArray[0]

                labelInverseMatrix[dependencyArray[2]][dependencyArray[1]] = dependencyArray[0]
                
                adjMatrix[dependencyArray[1]][dependencyArray[2]] = 1

                adjInverseMatrix[dependencyArray[2]][dependencyArray[1]] = 1


        print("Label matrix: ")
        print(labelMatrix)
        print(labelMatrix.shape)

        print("Adj matrix: ")
        print(adjMatrix)

        return labelMatrix, adjMatrix, labelInverseMatrix, adjInverseMatrix

    def getUnvisited(self):
        return self.dictionary.getUnvisited()


adjGenerator = AdjGenerator("train.en.out", batchSize=80)
label, adj, labelInverse, adjInverse = adjGenerator.generateMatrix()
for i in range (0, 1000):
    label2, adj2, labelInverse2, adjInverse2 = adjGenerator.generateMatrix()

print("Result: ")
print(label)
print(adj)
print(labelInverse)
print(adjInverse)
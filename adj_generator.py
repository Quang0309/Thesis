
from enum import Enum
import numpy as np
import torch


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
        self.anchorForEachBach = open("AnchorForEachBach.txt", "w")

    def to_sparse(self, x):
        """ converts dense tensor x to sparse format """
        x_typename = torch.typename(x).split('.')[-1]
        sparse_tensortype = getattr(torch.sparse, x_typename)

        indices = torch.nonzero(x)
        if len(indices.shape) == 0:  # if all elements are zeros
            return sparse_tensortype(*x.shape)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())

    def generateMatrix(self):
        arrayList = []
        arrayOfASentence = []
        firstLineOfABatch = True

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
            if (firstLineOfABatch and line != '(())'):
                text = line + '\n'
                self.anchorForEachBach.write(text)
                firstLineOfABatch = False

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

        # Case when eof but not reach batch size
        #   or, there's one empty line at the last of the file :) 
        if (arrayOfASentence != []): 
            print("Eof but not reach batch size...")
            arrayList.append(arrayOfASentence)
        arrayOfASentence = []
        
        numberOfSentence = len(arrayList)
        print("Number of sentences:")
        print(numberOfSentence)
        if (numberOfSentence == 0):
            print("EOF !!")
            return None, None, None, None

        # Shift the index of the word        
        for index, arrayOfASentence in enumerate(arrayList):
            shiftValue = index * self.numberOfWords
            for dependencyArray in arrayOfASentence:
                dependencyArray[1] += shiftValue
                dependencyArray[2] += shiftValue

        print("Array list after shifting value: ")
        print(arrayList)

        # Create the matrix
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




    def generateMatrixFromIDs(self, ids):
        arrayList = []
        arrayOfASentence = []
        firstLineOfABatch = True
        indexOfSentence = 1

        for line in self.inputFile:
            if (line in ['\n', '\r\n']):
                print("Empty line! End of a sentence.")
                if (arrayOfASentence != []): 
                    if (indexOfSentence in ids):
                        arrayList.append(arrayOfASentence)
                    
                    indexOfSentence += 1
                    
                arrayOfASentence = []
                if len(arrayList) == self.batchSize:
                    print("Completed a batch !!!")
                    break
            else:  # remove the line delimiter "\n" at the end of this line
                line = line.replace("\n", "")
            print(line)
            if (firstLineOfABatch and line != '(())'):
                text = line + '\n'
                self.anchorForEachBach.write(text)
                firstLineOfABatch = False

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

        # Case when eof but not reach batch size
        #   or, there's one empty line at the last of the file :) 
        if (arrayOfASentence != []): 
            print("Eof but not reach batch size...")
            if (indexOfSentence in ids):
                arrayList.append(arrayOfASentence)

        arrayOfASentence = []
        
        numberOfSentence = len(arrayList)
        print("Number of sentences:")
        print(numberOfSentence)
        # if (numberOfSentence == 0):
        #     print("EOF !!")
        #     return None, None, None, None

        # Shift the index of the word        
        for index, arrayOfASentence in enumerate(arrayList):
            shiftValue = index * self.numberOfWords
            for dependencyArray in arrayOfASentence:
                dependencyArray[1] += shiftValue
                dependencyArray[2] += shiftValue

        print("Array list after shifting value: ")
        print(arrayList)

        # Create the matrix
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

        self.inputFile.close()
        labelMatrix = torch.from_numpy(labelMatrix)
        labelMatrix = self.to_sparse(labelMatrix).long()

        adjMatrix = torch.from_numpy(adjMatrix)
        adjMatrix = self.to_sparse(adjMatrix).long()

        labelInverseMatrix = torch.from_numpy(labelInverseMatrix)
        labelInverseMatrix = self.to_sparse(labelInverseMatrix).long()

        adjInverseMatrix = torch.from_numpy(adjInverseMatrix)
        adjInverseMatrix = self.to_sparse(adjInverseMatrix).long()

        return labelMatrix, adjMatrix, labelInverseMatrix, adjInverseMatrix



adjGenerator = AdjGenerator("test.txt", numberOfWords=19, batchSize=80)
label, adj, labelInverse, adjInverse = adjGenerator.generateMatrixFromIDs([80, 100, 10000])
print("Result: ")
print(label)
print(adj)
print(labelInverse)
print(adjInverse)

# count = 1
# stop = False
# while (stop == False):
#     label2, adj2, labelInverse2, adjInverse2 = adjGenerator.generateMatrix()
#     if (label2 is None):
#         print("Done !!")
#         print("Count: ")
#         print(count)
#         stop = True
#     else: 
#         count += 1
#         print("Result 2:")
#         print(label2)
#         print(adj2)
#         print(labelInverse2)
#         print(adjInverse2)

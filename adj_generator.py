
from enum import Enum
import numpy as np


class Dictionary():
    def __init__(self):
        self.mydict = {
            "det": 1,
            "amod": 2,
            "nsubj": 3,
            "advmod": 4,
            "partmod": 5,
            "prep_in": 6,
            "ccomp": 7,
            "prt": 8,
            "dobj": 9,
            "prep_of": 10,
            "conj_and": 11,
            "nn": 12,
            "aux": 13,
            "xcomp": 14,
            "poss": 15,
            "conj_or": 16,
            "prep_during": 17,
            "tmod": 18,
            "root": 19,
            "mark": 20,
            "case": 21,
            "nmod:to": 22,
            "nmod:tmod": 23,
            "ref": 24,
            "acl:relcl": 25,
            "advcl": 26,
            "nmod:in": 27,
            "compound": 28,
            "nmod:with": 29,
            "cc": 30,
            "conj:and": 31,
            "conj:or": 32,
            "nummod": 33,
            "nmod:of": 34,
            "nmod:on": 35,
            "compound:prt": 36,
            "nmod:poss": 37,
            "nsubjpass": 38,
            "auxpass": 39,
            "nmod:by": 40,
            "nmod:from": 41,
            "cop": 42,
            "det:predet": 43,
            "dep": 44,
            "nmod:over": 45,
            "nmod:for": 46,
            "nmod:as": 47,
            "expl": 48,
            "appos": 49,
            "nsubj:xsubj": 50,
            "nmod:at": 51,
            "nmod:above": 52,
            "mwe": 53,
            "nmod:like": 54,
            "nmod:about": 55,
            "acl": 56,
            "nmod:inside": 57,
            "parataxis": 58,
            "nmod:out_of": 59,
            "neg": 60,
            "discourse": 61,
            "nmod:than": 62,
            "nmod:out": 63,
            "nmod:that": 64,
            "iobj": 65,
            "nmod:per": 66,
            "conj:but": 67,
        }
        self.unvisited = []

    def getDependencyValue(self, dependency):
        value = self.mydict.get(dependency)
        if (value == None):
            self.unvisited.append(dependency)
            #raise ValueError('Dependency does not exist!!!')
        else:
            return value

    def getUnvisited(self):
        return self.unvisited


class AdjGenerator:

    def __init__(self, fileName):
        self.fileName = fileName
        self.dictionary = Dictionary()

    def generateMatrix(self):
        inputFile = open(self.fileName, "r", encoding="utf8")
        arrayList = []
        arrayOfASentence = []
        maxNumberOfWords = 0

        for line in inputFile:
            if (line in ['\n', '\r\n']):
                print("Empty line! End of a sentence.")
                if (arrayOfASentence != []):
                    arrayList.append(arrayOfASentence)
                arrayOfASentence = []
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

                    if (firstNumber > maxNumberOfWords):
                        maxNumberOfWords = firstNumber
                    if (secondNumber > maxNumberOfWords):
                        maxNumberOfWords = secondNumber

        print("Max number of words in a sentence: ")
        print(maxNumberOfWords)

        print("Array list: ")
        if (arrayOfASentence != []):
            arrayList.append(arrayOfASentence)
        print(arrayList)

        # Shift the index of the word
        for index, arrayOfASentence in enumerate(arrayList):
            shiftValue = index * maxNumberOfWords
            for dependencyArray in arrayOfASentence:
                dependencyArray[1] += shiftValue
                dependencyArray[2] += shiftValue

        print("Array list after shifting value: ")
        print(arrayList)

        # Create the matrix
        numberOfSentence = len(arrayList)
        print("Number of sentences:")
        print(numberOfSentence)
        numberOfRows = numberOfSentence * maxNumberOfWords
        print("Number of rows:")
        print(numberOfRows)
        labelMatrix = np.zeros((numberOfRows, numberOfRows))
        adjMatrix = np.zeros((numberOfRows, numberOfRows))

        # Fill value to this matrix
        for index, arrayOfASentence in enumerate(arrayList):
            for dependencyArray in arrayOfASentence:
                labelMatrix[dependencyArray[1]
                            ][dependencyArray[2]] = dependencyArray[0]
                adjMatrix[dependencyArray[1]][dependencyArray[2]] = 1
        print("Label matrix: ")
        print(labelMatrix)

        print("Adj matrix: ")
        print(adjMatrix)

        inputFile.close()
        return labelMatrix, adjMatrix

    def getUnvisited(self):
        return self.dictionary.getUnvisited()


adjGenerator = AdjGenerator("train.en.out")
label, adj = adjGenerator.generateMatrix()
print("Result: ")
print(label)
print(adj)

print("Unvisited: ")
print(adjGenerator.getUnvisited())

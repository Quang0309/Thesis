
from enum import Enum
import numpy as np

class Dependency(Enum):
    det = 1
    amod = 2
    nsubj = 3
    advmod = 4
    partmod = 5
    prep_in = 6
    ccomp = 7
    prt = 8
    dobj = 9
    prep_of = 10
    conj_and = 11
    nn = 12
    aux = 13
    xcomp = 14
    poss = 15
    conj_or = 16
    prep_during = 17
    tmod = 18
    root = 19

class AdjGenerator:

    def __init__(self, fileName):
        self.fileName = fileName;
    
    def generateMatrix(self):
        inputFile = open(self.fileName, "r")    
        arrayList = []     
        arrayOfASentence = []
        maxNumberOfWords = 0 


        for line in inputFile:                           
            if (line in ['\n', '\r\n']):
                print("Empty line! End of a sentence.")
                            
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
                #print(character)            
                if (character == '('): # Dependency
                    dependency = line[0:index]
                    print("Dependency:")
                    print(dependency)

                elif (character == '-'):
                    start = index + 1
                elif (character == ',' or character == ')'):
                    end = index                    
                    if (isFirstNumber):
                        firstNumber = int(line[start:end])
                        print("Number: ")
                        print(firstNumber)        
                        isFirstNumber = False
                    else:
                        secondNumber = int(line[start:end])
                        print("Second number:")
                        print(secondNumber)   

                        if (dependency != 'root'):
                            # Creating an array with form (dependencyNumber, firstNumber, secondNumber) for this dependency        
                            dependencyArray = [Dependency[dependency].value, 
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
        arrayList.append(arrayOfASentence)
        print(arrayList)


        # Shift the index of the word                
        for index, arrayOfASentence in  enumerate(arrayList):                        
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
        for index, arrayOfASentence in  enumerate(arrayList):            
            for dependencyArray in arrayOfASentence:        
                labelMatrix[dependencyArray[1]][dependencyArray[2]] = dependencyArray[0]
                adjMatrix[dependencyArray[1]][dependencyArray[2]] = 1
        print("Label matrix: ")
        print(labelMatrix)

        print("Adj matrix: ")
        print(adjMatrix)

        inputFile.close()
        return labelMatrix, adjMatrix





adjGenerator = AdjGenerator("test.txt")
label, adj = adjGenerator.generateMatrix()
print("Result: ")
print(label)
print(adj)




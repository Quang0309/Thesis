import torch
import numpy as np
import os

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
                    # print(line)
                    self.mydict[line] = self.value
                    self.value += 1
            # print(self.mydict)    

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
        self.dictionaryOffsetFileName = "dictionary.txt"

        self.dictionary = Dictionary()
        self.numberOfWords = numberOfWords
        self.batchSize = batchSize

        dir_path = os.path.abspath(os.curdir)
        dir_path = dir_path +'/data-bin' '/tokenized.en-vi/'
        # print(dir_path)
        self.dir_path = dir_path
        self.fileName = self.dir_path + self.fileName
        self.dictionaryOffsetFileName = self.dir_path + self.dictionaryOffsetFileName

        # self.dir_path = fileName

        # self.inputFile = open(self.fileName, "r", encoding="utf8")
        self.anchorForEachBach = open("AnchorForEachBach.txt", "w")


        self.dictionaryOffsetArray = []
        f = open(self.dictionaryOffsetFileName, "r", encoding="utf8")
        for line in f: 
            line = line.strip()
            line = line.split(':')
            array = [int(line[0]), int(line[1])]
            # print(array)
            self.dictionaryOffsetArray.append(array)
        f.close()
        # print(self.dictionaryOffsetArray[0])
        # print(self.dictionaryOffsetArray[1])
        # print(self.dictionaryOffsetArray[2])

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

    def generateTensorsFromIDs(self, ids, numberOfWordsPerSentence, batchSize):
        self.inputFile = open(self.fileName, "r", encoding="utf8")
        self.numberOfWords = numberOfWordsPerSentence
        self.batchSize = batchSize
        arrayList = []
        

        for id in ids:            
            arrayOfASentence = []

            offsetMapping = self.dictionaryOffsetArray[id]        
            startOffset = offsetMapping[0] # start offset of the file pointer
            length = offsetMapping[1] # length of this sentence

            self.inputFile.seek(startOffset)
            lines = self.inputFile.read(length).split('\n')
            # print("input: ")
            # print(lines)

            if (lines == '(())'):                
                arrayList.append([])
            else:
                for line in lines:
                    # print(line)
                    start = 0
                    end = 0
                    isFirstNumber = True
                    firstNumber = -1
                    secondNumber = -1
                    dependency = ''
                    for index, character in enumerate(line):                        
                        if (character == '('):  # Dependency
                            if (line[index + 1] != '('):  # check the char next to the '('
                                dependency = line[0:index]
                                # print("Dependency:")
                                # print(dependency)
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
                                # print("Number: ")
                                # print(firstNumber)
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
                                # print("Second number:")
                                # print(secondNumber)

                                if (dependency != 'root'):
                                    if (firstNumber <= self.numberOfWords and secondNumber <= self.numberOfWords):
                                    # Creating an array with form (dependencyNumber, firstNumber, secondNumber) for this dependency
                                        dependencyArray = [self.dictionary.getDependencyValue(dependency),
                                               firstNumber - 1,
                                               secondNumber - 1]
                                        # print("array: ")
                                        # print(dependencyArray)
                                        arrayOfASentence.append(dependencyArray)                    
                            
            # print("End !!")
            # print(arrayOfASentence)
            arrayList.append(arrayOfASentence)

        # print("Array list:")
        # print(arrayList)
        
        # numberOfSentence = len(arrayList)
        # print("Number of sentences:")
        # print(numberOfSentence)

        # Shift the index of the word        
        for index, arrayOfASentence in enumerate(arrayList):
            shiftValue = index * self.numberOfWords
            for dependencyArray in arrayOfASentence:
                dependencyArray[1] += shiftValue
                dependencyArray[2] += shiftValue
        

        # print("Array list after shifting value: ")
        # print(arrayList)

        # Create the matrix
        numberOfRows = self.numberOfWords * self.batchSize
        # print("Number of rows:")
        # print(numberOfRows)
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


        # print("Label matrix: ")
        # print(labelMatrix)
        # print(labelMatrix.shape)

        # print("Adj matrix: ")
        # print(adjMatrix)

        self.inputFile.close()
        labelMatrix = torch.from_numpy(labelMatrix)
        labelMatrix = self.to_sparse(labelMatrix).long()

        adjMatrix = torch.from_numpy(adjMatrix)
        adjMatrix = self.to_sparse(adjMatrix).long()

        labelInverseMatrix = torch.from_numpy(labelInverseMatrix)
        labelInverseMatrix = self.to_sparse(labelInverseMatrix).long()

        adjInverseMatrix = torch.from_numpy(adjInverseMatrix)
        adjInverseMatrix = self.to_sparse(adjInverseMatrix).long()

        return labelMatrix.cuda(), adjMatrix.cuda(), labelInverseMatrix.cuda(), adjInverseMatrix.cuda()
        # return labelMatrix, adjMatrix, labelInverseMatrix, adjInverseMatrix


    def generateDictionaryMapping(self):
        self.dictionaryFile = open(self.dictionaryOffsetFileName, "w", encoding="utf8")
        inputFile = open(self.fileName, "r", encoding="utf8")
        isNewSentence = True

        count = 0
        numberOfSentence = 0
        startOffsetForEachSentence = 0
        sentenceLength = 0
        inputFile.seek(0)

        line = inputFile.readline()
        while (line):
            if (line in ['\n', '\r\n']):                
                isNewSentence = True
                # print("length of prev sentence: ")
                # print(sentenceLength)
                length = ':' + str(sentenceLength) + '\n'
                self.dictionaryFile.write(length)
                sentenceLength = 0
            else:  # remove the line delimiter "\n" at the end of this line                
                line = line.replace("\n", "")                   
                if (line == '(())'):
                    numberOfSentence += 1
                    # print("A sentence > 50")
                    # print(line)   
                    sentenceLength = len(line) + 1 # we have remove the \n of this line
                    # print("length of sentence: ")                    
                    # print(sentenceLength)                 
                    offset = str(startOffsetForEachSentence) + ':' + str(sentenceLength) + '\n'
                    self.dictionaryFile.write(offset)
                    
                elif (isNewSentence):                    
                    sentenceLength = len(line) + 1 # we have remove the \n of this line
                    numberOfSentence += 1
                    isNewSentence = False
                    # print("A sentence")
                    # print(line)
                    offset = str(startOffsetForEachSentence)
                    self.dictionaryFile.write(offset)
                else:      
                    sentenceLength += len(line) + 1
                
                    
            startOffsetForEachSentence = inputFile.tell()            
            inputFile.seek(startOffsetForEachSentence)
            line = inputFile.readline()
            # count += 1
            # if (count > 300):
            #     break


        # inputFile.seek(720)
        # print("Test: ")
        # s= inputFile.read(871)
        # print(s)      

        inputFile.close()
        # print("Number of sentence")
        # print(numberOfSentence)

    def test(self):
        dictionaryFile = open(self.dictionaryOffsetFileName, "r", encoding="utf8")
        inputFile = open(self.fileName, "r", encoding="utf8")
        outputFile = open(self.dir_path + "test_dict.txt", "w", encoding="utf8")
        count = 0
        for line in dictionaryFile:
            line = line.strip()            
            line = line.split(':')            
            startOffSet = int(line[0])
            sentenceLength = int(line[1])

            inputFile.seek(startOffSet)
            s = inputFile.read(sentenceLength) + '\n'
            outputFile.write(s)

            # count += 1
            # if (count > 30):
            #     break
    
    def readSentence(self, startOffset, length):
        inputFile = open(self.fileName, "r", encoding="utf8")
        inputFile.seek(startOffset)
        s = inputFile.read(length)
        print(s)
        # print(self.dictionaryOffsetArray[60])
        inputFile.close()
        return s
    
    def readSentenceID(self, id):
        offsetMapping = self.dictionaryOffsetArray[id]        
        startOffset = offsetMapping[0] # start offset of the file pointer
        length = offsetMapping[1] # length of this sentence
        inputFile = open(self.fileName, "r", encoding="utf8")
        inputFile.seek(startOffset)
        s = inputFile.read(length)
        print("Read sentence id: ")
        print(s)
        inputFile.close()



        
adj = AdjGenerator("train.en.out")
# adj.generateTensorsFromIDs([1, 0, 2, 60], 50, 4)

# ids = []
# for i in range (50000,50160, 2):
#     ids.append(i)
# adj.generateTensorsFromIDs(ids, 50, 80)


# adj.readSentence(25405, 19)
# adj.readSentence(25174,219)
# adj.readSentence(45822, 5)
# adj.readSentence(45828, 574)
adj.readSentenceID(101)

# adj.readSentenceID(145403)

# adj.generateDictionaryMapping()
# adj.test()

        
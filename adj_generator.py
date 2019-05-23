# import torch
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
        self.dictionary = Dictionary()
        self.numberOfWords = numberOfWords
        self.batchSize = batchSize

        # dir_path = os.path.abspath(os.curdir)
        # dir_path = dir_path +'/data-bin' '/tokenized.en-vi/' + self.fileName
        # print(dir_path)
        # self.dir_path = dir_path
        self.dir_path = fileName

        # self.inputFile = open(self.fileName, "r", encoding="utf8")
        self.anchorForEachBach = open("AnchorForEachBach.txt", "w")


        self.dictionaryOffsetArray = []
        f = open("dictionary.txt", "r", encoding="utf8")
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
        self.inputFile = open(self.dir_path, "r", encoding="utf8")
        self.numberOfWords = numberOfWordsPerSentence
        self.batchSize = batchSize
        arrayList = []
        arrayOfIDs = []
        arrayOfASentence = []
        firstLineOfABatch = True
        indexOfSentence = 0
        isSentenceWithRoot = False
        

        for line in self.inputFile:
            #if(indexOfSentence in ids):
                #print('long-> ' + line)
            if (line in ['\n', '\r\n']):
                #print("Empty line! End of a sentence.")
                if (arrayOfASentence != []): 
                    if (indexOfSentence in ids):
                        arrayOfIDs.append(indexOfSentence)
                        arrayList.append(arrayOfASentence)                
                    indexOfSentence += 1
                elif (isSentenceWithRoot): # case when sentence has only Root(bla, bla)
                    isSentenceWithRoot = False  
                    if (indexOfSentence in ids):
                        arrayOfIDs.append(indexOfSentence)
                        arrayList.append(arrayOfASentence)                
                    indexOfSentence += 1

                arrayOfASentence = []
                if len(arrayList) == self.batchSize:
                 #   print("Completed a batch !!!")
                    break
            else:  # remove the line delimiter "\n" at the end of this line
                line = line.replace("\n", "")

            # print(line)

            if (line == '(())'):
                if (indexOfSentence in ids):
                    arrayOfIDs.append(indexOfSentence)
                    arrayList.append([])
                indexOfSentence += 1    

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
                        else:
                            isSentenceWithRoot = True

        # Case when eof but not reach batch size
        #   or, there's one empty line at the last of the file :) 
        if (arrayOfASentence != []): 
            # print("Eof but not reach batch size...")
            if (indexOfSentence in ids):
                arrayOfIDs.append(indexOfSentence)
                arrayList.append(arrayOfASentence)

        arrayOfASentence = []
        
        # numberOfSentence = len(arrayList)
        # print("Number of sentences:")
        # print(numberOfSentence)

        # if (numberOfSentence == 0):
        #     print("EOF !!")
        #     return None, None, None, None

        # Rearrange in the order of ids
        resultArrayList = []
        # print("Array of ids")
        # print(ids)
        # print("ids unordered")
        # print(arrayOfIDs)

        for id in ids:
            for index, value in enumerate(arrayOfIDs):
                if (value == id):
                    resultArrayList.append(arrayList[index])

        # Shift the index of the word        
        for index, arrayOfASentence in enumerate(resultArrayList):
            shiftValue = index * self.numberOfWords
            for dependencyArray in arrayOfASentence:
                dependencyArray[1] += shiftValue
                dependencyArray[2] += shiftValue
        

        #print("Array list after shifting value: ")
        #print(resultArrayList)

        # Create the matrix
        numberOfRows = self.numberOfWords * self.batchSize
        # print("Number of rows:")
        # print(numberOfRows)
        labelMatrix = np.zeros((numberOfRows, numberOfRows))
        adjMatrix = np.zeros((numberOfRows, numberOfRows))
        labelInverseMatrix = np.zeros((numberOfRows, numberOfRows))
        adjInverseMatrix = np.zeros((numberOfRows, numberOfRows))

        # Fill value to this matrix
        for index, arrayOfASentence in enumerate(resultArrayList):
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
        # labelMatrix = torch.from_numpy(labelMatrix)
        # labelMatrix = self.to_sparse(labelMatrix).long()

        # adjMatrix = torch.from_numpy(adjMatrix)
        # adjMatrix = self.to_sparse(adjMatrix).long()

        # labelInverseMatrix = torch.from_numpy(labelInverseMatrix)
        # labelInverseMatrix = self.to_sparse(labelInverseMatrix).long()

        # adjInverseMatrix = torch.from_numpy(adjInverseMatrix)
        # adjInverseMatrix = self.to_sparse(adjInverseMatrix).long()

        # return labelMatrix.cuda(), adjMatrix.cuda(), labelInverseMatrix.cuda(), adjInverseMatrix.cuda()
        return labelMatrix, adjMatrix, labelInverseMatrix, adjInverseMatrix


    def generateDictionaryMapping(self):
        self.dictionaryFile = open("dictionary.txt", "w")
        inputFile = open(self.dir_path, "r", encoding="utf8")
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
        dictionaryFile = open("dictionary.txt", "r", encoding="utf8")
        inputFile = open("train.en.out", "r", encoding="utf8")
        outputFile = open("test_dict.txt", "w", encoding="utf8")
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

        
adj = AdjGenerator("train.en.out")
# adj.generateDictionaryMapping()
# adj.test()
        
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset

import os 


def collate(
    samples, pad_idx, eos_idx,  adj_generator,  left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    print("Here")
    print(id)
    adjTensor, labelTensor, adjInverseTensor, labelInverseTensor = adj_generator.generateTensorsFromIDs(id)
    print(adjTensor)
    

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        data_path, split,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=False, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.data_path = data_path
        self.split = split

        name = ''
        if (split == 'train'):
            name = 'train.en.out'
        elif (split == 'valid'):
            name = 'valid.en.out'
        self.adj_generator = AdjGenerator(name, numberOfWords=19, batchSize=80)

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, adj_generator=self.adj_generator
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)


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

        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = dir_path + '/' + self.fileName
        # print(dir_path)

        # self.inputFile = open(self.fileName, "r", encoding="utf8")
        self.inputFile = open(dir_path, "r", encoding="utf8")
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

    def generateTensorsFromIDs(self, ids):
        arrayList = []
        arrayOfIDs = []
        arrayOfASentence = []
        firstLineOfABatch = True
        indexOfSentence = 0
        

        for line in self.inputFile:
            if (line in ['\n', '\r\n']):
                #print("Empty line! End of a sentence.")
                if (arrayOfASentence != []): 
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
            #print(line)
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
                            # Creating an array with form (dependencyNumber, firstNumber, secondNumber) for this dependency
                            dependencyArray = [self.dictionary.getDependencyValue(dependency),
                                               firstNumber - 1,
                                               secondNumber - 1]
                            # print("array: ")
                            # print(dependencyArray)
                            arrayOfASentence.append(dependencyArray)                    

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

        print("Array list after shifting value: ")
        print(resultArrayList)

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
        labelMatrix = torch.from_numpy(labelMatrix)
        labelMatrix = self.to_sparse(labelMatrix).long()

        adjMatrix = torch.from_numpy(adjMatrix)
        adjMatrix = self.to_sparse(adjMatrix).long()

        labelInverseMatrix = torch.from_numpy(labelInverseMatrix)
        labelInverseMatrix = self.to_sparse(labelInverseMatrix).long()

        adjInverseMatrix = torch.from_numpy(adjInverseMatrix)
        adjInverseMatrix = self.to_sparse(adjInverseMatrix).long()

        return labelMatrix, adjMatrix, labelInverseMatrix, adjInverseMatrix
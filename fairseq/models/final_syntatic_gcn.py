import torch
import torch.nn as nn
from typing import Optional

def segment_sum(embeddings, segment_ids):
    if (embeddings.shape[0] != segment_ids.shape[0]):
        raise ValueError('The first dimension of embeddings must equal to the segment_ids size.')

    # This tensor will have size equal to the first tensor in our data tensor
    tensor_return = torch.zeros_like(embeddings[:, :].clone().split(1)[0])
    #print(tensor_return)

    # This tensor will have size equal to the first tensor in our data tensor, remains unchanged !
    zero_tensor = torch.zeros_like(embeddings[:, :].clone().split(1)[0])

    first_time = True
    sum = 0
    index = 0
    start = 0
    prev_index = 0

    segment_id = segment_ids.split(1)[0].item()
    #print(segment_id)
    
    #print("--- segment_ids.split(1): ")
    #print(segment_ids.split(1))

    for id in segment_ids.split(1):
        #print(' ----   Iteration width id: -----')
        #print(id.item())
        if (id.item() == segment_id):
            #print("equal")
            
            
            # Ids not given in our segment_ids will give tensor with 0 value
            for missing_id in range(prev_index, id.item()):
                #print("missing_id: ")
                #print(missing_id)
                if (first_time):
                    # Initially, this tensor has 0 values in it, we must re-assign it
                    tensor_return = zero_tensor.clone()
                    first_time = False
                else:
                    tensor_return = torch.cat((tensor_return, zero_tensor), 0)
            
            prev_index = id.item()
        
        else:                
            # found diff index

        
            # Get a list of tensors from starting row to index row from our data tensor
            b = embeddings[start:index, :].clone()
        
            # Create a tensor sum_t that has 0 values and shape like the first tensor
            sum_t = torch.zeros_like(b.split(1)[0])
        
            # Sum the value of each tensor to sum_t
            for t in b.split(1):
                #print(t)
                sum_t += t 
        
            #print("t: ")
            #print(sum_t)
            
            
            if (first_time):
                # Initially, this tensor has 0 values in it, we must re-assign it
                tensor_return = sum_t.clone()
                first_time = False
            else:
                tensor_return = torch.cat((tensor_return, sum_t), 0)
                
                
        
            if id.item() - prev_index > 1:
                # Ids not given in our segment_ids will give tensor with 0 value
                prev_missing_id = -1

                for missing_id in range(prev_index + 1, id.item()):
                    #print("missing_id: ")
                    #print(missing_id)
                    if (prev_missing_id != missing_id):
                        if (first_time):
                            # Initially, this tensor has 0 values in it, we must re-assign it
                            tensor_return = zero_tensor.clone()
                            first_time = False
                        else:
                            tensor_return = torch.cat((tensor_return, zero_tensor), 0)
            
            prev_index = id.item()
        
        
            segment_id = id.item()          
        
            start = index  
    
        index += 1
    

    
    # Get a list of tensors from starting row to index row from our data tensor
    b = embeddings[start:index, :].clone()

    # Create a tensor sum_t that has 0 values and shape like the first tensor
    sum_t = torch.zeros_like(b.split(1)[0])
    for t in b.split(1):
        #print(t)
        sum_t += t
        
    #print("t: ")
    #print(sum_t)
    
    if (first_time):
        tensor_return = sum_t.clone()
        first_time = False
    else:
        tensor_return = torch.cat((tensor_return, sum_t), 0)


    # Ids not given in our segment_ids will give tensor with 0 value
    prev_missing_id = -1
    for missing_id in range(prev_index, index - 1):
        #print("missing_id: ")
        #print(missing_id)
        if (prev_missing_id != missing_id):
            prev_missing_id = missing_id
            if (first_time):
                # Initially, this tensor has 0 values in it, we must re-assign it
                tensor_return = zero_tensor.clone()
                first_time = False
            else:
                tensor_return = torch.cat((tensor_return, zero_tensor), 0)
    #print("    Tensor return: ")
    #print(tensor_return)
    return tensor_return


def embedding_lookup_sparse(embeddings, sparse_ids, sparse_weights):
  #  print("Embedding lookup sparse")
    
  #  print("--- Embeddings:")
  #  print(embeddings.size())
    
  #  print("--- sparse_ids:")
  #  print(sparse_ids)
  #  print("sparse_ids size:")
  #  print(sparse_ids.size())
  #  print("sparse_ids values: ")
  #  print(sparse_ids._values())
  #  print("sparse_ids values size:")
  #  print(sparse_ids._values().size())
    
  #  print("--- sparse_weights:")
  #  print(sparse_weights.size())
  #  print("sparse_weights values: ")
  #  print(sparse_weights._values())
  #  print("sparse_weights values size:") 
  #  print(sparse_weights._values().size())
    
    if (sparse_ids.size() != sparse_weights.size()):
        raise ValueError("Shape of sparse_ids and sparse_weights are incompatible")
    
    with torch.no_grad():
        a = sparse_ids._indices()
    #    print("--- sparse_ids indices: ")
    #    print(a)    
    
        segment_ids = sparse_ids._indices()[0] # get the tensor containing all the rows
    #    print("--- segment_ids:")
    #    print(segment_ids)
    
        ids = sparse_ids._values()
    #    print("--- ids = sparse_ids.values(): ")
    #    print(ids)
    
        ids, idx = torch.unique(ids, sorted=True, return_inverse=True)
    #    print("--- After unique(ids): ")
    #    print("ids: ")
    #    print(ids)
    #    print("idx: ")
    #    print(idx)
    
        embeddings = embedding_lookup(embeddings, ids)
    #    print("--- Embeddings after embedding_lookup: ")
    #    print(embeddings)
    #    print(embeddings.size())
    
        weights = sparse_weights._values().unsqueeze(-1)
        #print("Weights: ")
        #print(weights)
        #print(weights.size())
    
    
        origin_size = embeddings.size()
        #print("--- origin size:")
        #print(origin_size)
    
        embeddings = embeddings[idx] # equivalent to tf.gather     
        #print("--- Embeddings after embeddings[idx] ")
        #print(embeddings)    
        #print(embeddings.size())
    
    
        # Reshape weights to allow broadcast
#     ones = array_ops.fill(
#           array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
#     x = torch.ones(embeddings.dim() - 1)
#     print(x.size())
#     print(x)

#     x = x.unsqueeze(0)
#     print(x.size())
#     print(x)

#     ones = torch.ones(x.size())
#     print("--- ones: ")
#     print(ones.size())
#     print(ones)
    
#     bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],0)
#     print("Weigths size: ")
#     print(weights.size())
#     bcast_weights_shape = torch.cat((torch.zeros(weights.size()), ones), 0)
    
#     orig_weights_shape = weights.size()
#     print("--- original weigths shape: ")
#     print(orig_weigths_shape)
    
#     weights = array_ops.reshape(weights, bcast_weights_shape)
    
    
#     embeddings = math_ops.segment_sum(embeddings, segment_ids)  
    #print("--- embeddings size: ")
    #print(embeddings.size())
    #print("--- weights size: ")
    #print(weights.size())
    
#     torch.reshape(weights, (embeddings.size()[0], -1))
#     weights = weights.unsqueeze(-1)
#     print("new weights size: ")
#     print(weights.size())
        weights = weights.float()
        embeddings *= weights
        #print(" embeddings after multiplication: ")
        #print(embeddings.size())
    
        result = segment_sum(embeddings, segment_ids)
        #print("Result: ")
        #print(result)
        #print(result.size())
        embeddings = torch.zeros((sparse_ids.size()[0], origin_size[1])) # 2720 * 256
        embeddings.new_tensor(result)

        #print("--- Return embeddings...")
        #print(embeddings.size())
        return embeddings



def sparse_dropout_mask(variable: torch.sparse.FloatTensor,
                        keep_prob: float,
                        train_mode: torch.Tensor) -> torch.sparse.FloatTensor:
    """Performs dropout on a sparse tensor, depending on mode. """

    #shape = tf.shape(variable.values)
    shape = variable._values().size()
    #print(shape)
    #print(variable.values)
    #with tf.variable_scope("dropout"):
    if keep_prob == 1.0:
        #return tf.fill(shape, True)
        v = torch.empty(shape)
        return v.fill_(True)

    keep_prob = torch.where(train_mode, torch.tensor(keep_prob), torch.tensor(1.0))

    #probs = tf.random_uniform(shape) + keep_prob
    t = torch.empty(shape)
    probs = torch.nn.init.uniform_(t) + keep_prob

    #return tf.cast(tf.floor(probs), dtype=tf.bool)
    return torch.floor(probs).byte()

def sparse_retain(tensor,masked_select):
    masked_select = torch.squeeze(masked_select)
    c = tensor._values()
    c = torch.squeeze(c)
    c = c[masked_select] # remove value with dropout
    d = tensor._indices()
    d = d.t()
    d = d[masked_select] # remove indices of values with dropout
    t = torch.sparse_coo_tensor(d.t(), c,tensor.size()) # construct a new sparse tensor
    return t


#import torch

#n = torch.tensor([[[1,1],[2,2],[3,3],[4,4]],[[11,11],[12,12],[13,13],[14,14]],
#                          [[21,21],[22,22],[23,23],[24,24]]])

def embedding_lookup(embeddings, indices):
    #print(indices.size())
    #print(embeddings.index_select(0, indices.view(-1)))
    with torch.no_grad():
        indices.cuda()
        return embeddings.index_select(0, indices.view(-1)).view(*(indices.size() + embeddings.size()[1:]))

#print(n)

#ids = torch.tensor([0,2])

#print("Embedding lookup of n: ")
#print(embedding_lookup(n, ids))

def sparse_fill_empty_rows(t, default_value):
    #print(type(default_value))
    row = t.size()[0]
    indices = t._indices().t()
    #print(indices.type())
    mdict = [item[0].item() for item in indices]
    notIn = [([i,0]) for i in range(0,row) if i not in mdict]
    b = torch.tensor(notIn,dtype = torch.long)
    indices = torch.cat((indices,b))
    zeros = torch.zeros([len(notIn)],dtype = torch.long)
    zeros.fill_(default_value)
    if type(default_value) == float:
        #print("float")
        zeros = zeros.float()
    #print(t._values().type())
    #print(zeros.type())
    values = torch.cat((t._values(),zeros))
    #else:
    #    values = torch.cat((t._values().float(),zeros))
    ten = torch.sparse_coo_tensor(indices.t(), values,t.size())
    return ten


def sparse_fill_empty_rows_V2(t, default_value):
    row = t.size()[0]
    indices = t._indices().t()
    values = t._values()
    new_indices = []
    new_values = []
    i = 0
    #if(indices[0] is not None):
    #if not indices:
    if len(indices != 0):
        while(i!=indices[0][0].item()):
            new_indices.append([i,0])
            new_values.append(default_value)
            i = i+1  
        for index, item in enumerate(indices):
            if(item[0].item()==i):
           
                new_indices.append([item[0].item(),item[1].item()])
                new_values.append(values[index])         
            else:
                i = i + 1
                while (item[0].item() != i):
                    new_indices.append([i,0])
                    new_values.append(default_value)
                    i = i+1  
                new_indices.append([item[0].item(),item[1].item()])
                new_values.append(values[index])          
        i = i + 1   
    while(i<row):
        new_indices.append([i,0])
        new_values.append(default_value)
        i = i + 1  
    new_indices_tensor = torch.tensor(new_indices)
    ten = torch.sparse_coo_tensor(new_indices_tensor.t(), new_values,t.size())
    return ten 
#71-75
#h = torch.matmul(inputs2d, self.w)
#h = torch.sparse.mm(adj, h)
#label_pads = sparse_fill_empty_rows(labels,0)
#labels_weights = sparse_fill_empty_rows(adj, 0.)


#test
#i = torch.tensor([[0,0],[1,2],[0,1],[9,2]])
#v = torch.tensor([1,2,3,10])
#t = torch.sparse_coo_tensor(i.t(), v,[10,10])
#ten = sparse_fill_empty_rows(t,0)
#print(ten)

#i = torch.tensor([[0,0],[1,1],[2,2],[3,3]])
#v = torch.tensor([1,2,3,4])
#t = torch.sparse_coo_tensor(i.t(), v,[5,5])
#print(t)
#b = torch.tensor(True)
#prob = sparse_dropout_mask(t,0.5,b)
#prob = torch.squeeze(prob)
#print(prob)
#tensor = sparse_retain(t,prob)
#print(tensor)


#c = t._values()
#c = torch.squeeze(c)
#c = c[prob]


#d = t._indices()
#d = [i for i in d if ]
#d = d.t()
#print(d)
#prob = torch.squeeze(prob)
#print(prob)
#d = torch.masked_select(d,prob)
#d = d[prob]
#t = torch.sparse_coo_tensor(d.t(), c,t.size())
#print(t)

def element_wise_mul(sparse_matrix, dense_matrix):
    indices = sparse_matrix._indices().t()
    values = sparse_matrix._values()
    for i in range(indices.size()[0]):
        index = indices[i]
        values[i] = values[i] * dense_matrix[0][index[1]]
    return torch.sparse_coo_tensor(indices.t(), values,sparse_matrix.size())

class DirectedGCN:
    def __init__(self, layer_size: int, num_labels: int, train_mode,
                 dropout_keep_p: float = 0.8,
                 edge_dropout_keep_p: float=0.8,
                 residual: Optional[bool] = True,
                 name: str='gcn'):
        self.layer_size = layer_size
        self.num_labels = num_labels
        self.train_mode = train_mode

        self.dropout_keep_p = dropout_keep_p
        self.edge_dropout_keep_p = edge_dropout_keep_p
        self.residual = residual
        self.name = name
        self._create_weight_matrices()
    def __call__(self, inputs, adj, labels, adj_inv, labels_inv):
        # graph convolution, heads to dependents ("out")
        # gates are applied through the adjacency matrix values

        # apply sparse dropout
        
        # adj = adj.cuda()
        # labels = labels.cuda()
        # adj_inv = adj_inv.cuda()
        # labels_inv = labels_inv.cuda()
        state_dim = inputs.size()[2]
        inputs2d = torch.reshape(inputs, [-1, state_dim])

        to_retain = sparse_dropout_mask(
            adj, self.edge_dropout_keep_p, self.train_mode)
        to_retain = torch.squeeze(to_retain)

        adj = sparse_retain(adj, to_retain).cuda()
        labels = sparse_retain(labels, to_retain).cuda()
        
        # apply gates        
        gates = torch.matmul(inputs2d, self.w_gate)
    
        #adj *= torch.t(gates)
        adj = element_wise_mul(adj,torch.t(gates))
        gates_bias = torch.squeeze(embedding_lookup(self.b_gate, labels._values()))
        #print(gates_bias)
        #print(adj._values())
        #temp = adj._values().float() + gates_bias
        #print(temp)
        sigmoid = nn.Sigmoid()
        values = sigmoid(adj._values().float() + gates_bias)

        # dropout scaling
        values /= torch.where(self.train_mode, torch.tensor(self.edge_dropout_keep_p), torch.tensor(1.0))

        adj = torch.sparse_coo_tensor(adj._indices(), values, adj.size())
        
        # graph convolution, heads to dependents ("out")
        h = torch.matmul(inputs2d, self.w)
        
        #h = tf.sparse_tensor_dense_matmul(adj, h)
        # print("adj size:")
        # print(adj.size())
        # print("h size:")
        # print(h.size())
        with torch.no_grad():
            h = torch.sparse.mm(adj, h).cuda()
        #h = torch.spmm(adj, h).cuda()
        #labels_pad, _ = tf.sparse_fill_empty_rows(labels, 0)
        #labels_weights, _ = tf.sparse_fill_empty_rows(adj, 0.)
        #print(labels)
        #print(adj)
        labels_pad = sparse_fill_empty_rows_V2(labels,0).cuda()
        labels_weights = sparse_fill_empty_rows_V2(adj, 0.).cuda()
        #print(self.b.size())
        # print("Labels _ pad : ")
        # print(labels_pad)
        # print(labels_pad.size())
        # print("Labels _ weigths : ")
        # print(labels_weights)
        # print(labels_weights.size())
        labels = embedding_lookup_sparse(self.b, labels_pad, labels_weights).cuda()
        # print("Labels: ")
        # print(labels.size())
        # print("b: ")
        # print(self.b.size())

        h = h + labels
        h = torch.reshape(h,inputs.size())
        #h = tf.reshape(h, tf.shape(inputs))

        # graph convolution, dependents to heads ("in")
        # gates are applied through the adjacency matrix values

        # apply sparse dropout
        to_retain_inv = sparse_dropout_mask(
            adj_inv, self.edge_dropout_keep_p, self.train_mode)
        to_retain_inv = torch.squeeze(to_retain_inv)

        adj_inv = sparse_retain(adj_inv, to_retain_inv).cuda()
        labels_inv = sparse_retain(labels_inv, to_retain_inv).cuda()

        # apply gates
        gates_inv = torch.matmul(inputs2d, self.w_gate_inv)
        #adj_inv *= torch.t(gates_inv)
        adj_inv = element_wise_mul(adj_inv,torch.t(gates_inv))
        gates_inv_bias = torch.squeeze(embedding_lookup(self.b_gate_inv, labels_inv._values()))
        values_inv = sigmoid(adj_inv._values().float() + gates_inv_bias)

        # dropout scaling
        values_inv /= torch.where(self.train_mode, torch.tensor(self.edge_dropout_keep_p), torch.tensor(1.0)).cuda()
        adj_inv = torch.sparse_coo_tensor(adj_inv._indices(), values_inv, adj_inv.size()).cuda()


        # graph convolution, dependents to heads ("in")
        
        h_inv = torch.matmul(inputs2d, self.w_inv).cuda()

        # h_inv = tf.sparse_tensor_dense_matmul(adj_inv, h_inv)
        with torch.no_grad():
            h_inv = torch.sparse.mm(adj_inv, h_inv)
        #h_inv = torch.spmm(adj_inv, h_inv)
        # labels_inv_pad, _ = tf.sparse_fill_empty_rows(labels_inv, 0)
        labels_inv_pad = sparse_fill_empty_rows_V2(labels_inv,0).cuda()

        # labels_inv_weights, _ = tf.sparse_fill_empty_rows(adj_inv, 0.)
        labels_inv_weights = sparse_fill_empty_rows_V2(adj_inv,0).cuda()

        labels_inv = embedding_lookup_sparse(self.b_inv, labels_inv_pad, labels_inv_weights,).cuda()

        h_inv = h_inv + labels_inv
        h_inv = torch.reshape(h_inv, inputs.size())

        # graph convolution, loops
        gates_loop = sigmoid(torch.matmul(inputs2d, self.w_gate_loop) + self.b_gate_loop)
        h_loop = torch.matmul(inputs2d, self.w_loop) + self.b_loop

        h_loop = h_loop * gates_loop
        h_loop = torch.reshape(h_loop, inputs.size())

        # final result is the sum of those (with residual connection to inputs)        
        relu = nn.ReLU()
        h = relu(h + h_inv + h_loop)

        if self.residual:
            print("GCN has residual connection")           
            return h + inputs
        else:
            print("GCN without residual connection")
            return h

    def _create_weight_matrices(self):
        """Creates all GCN weight matrices."""
        self.w = torch.empty(self.layer_size,self.layer_size,requires_grad=True).cuda()
        nn.init.normal_(self.w, std = 0.01)
       
        self.w_inv = torch.empty(self.layer_size,self.layer_size,requires_grad=True).cuda()
        nn.init.normal_(self.w_inv, std = 0.01)
        
        self.w_loop = torch.empty(self.layer_size,self.layer_size,requires_grad=True).cuda()
        nn.init.normal_(self.w_loop, std = 0.01)
       
        self.w_gate = torch.empty(self.layer_size,1,requires_grad=True).cuda()
        nn.init.normal_(self.w_gate, std = 0.01)
        
        self.w_gate_inv = torch.empty(self.layer_size,1,requires_grad=True).cuda()
        nn.init.normal_(self.w_gate_inv, std = 0.01)
        
        self.w_gate_loop = torch.empty(self.layer_size,1,requires_grad=True).cuda()
        nn.init.normal_(self.w_gate_loop, std = 0.01)
        
        self.b_gate = torch.empty(self.num_labels,requires_grad=True).cuda()
        nn.init.normal_(self.b_gate,mean = 0.0, std = 0.01)
        
        self.b_gate_inv = torch.empty(self.num_labels,requires_grad=True).cuda()
        nn.init.normal_(self.b_gate_inv,mean = 0.0, std = 0.01)
        
        self.b_gate_loop= torch.tensor([1.],requires_grad=True).cuda()
       
        self.b = torch.empty(self.num_labels,self.layer_size,requires_grad=True).cuda()
        nn.init.normal_(self.b, std = 0.01)
       
        self.b_inv = torch.empty(self.num_labels,self.layer_size,requires_grad=True).cuda()
        nn.init.normal_(self.b_inv, std = 0.01)
        
        self.b_loop = torch.empty(self.layer_size,requires_grad=True).cuda()
        nn.init.normal_(self.b_loop, std = 0.01)
        
#layer_size = 5
#num_labels = 3
#train_mode = torch.tensor(True)
#i = torch.tensor([[0,0],[0,1],[1,1],[2,2],[2,4],[3,3],[4,3],[4,4]])
#v = torch.tensor([1,1,1,1,1,1,1,1])
#adj = torch.sparse_coo_tensor(i.t(), v,[5,5])
#labels = torch.sparse_coo_tensor(i.t(), v,[5,5])
#i = torch.tensor([[0,0],[1,0],[1,1],[2,2],[3,3],[3,4],[4,4],[4,2]])
#adj_inv = torch.sparse_coo_tensor(i.t(), v,[5,5])
#labels_inv = torch.sparse_coo_tensor(i.t(), v,[5,5])
#gcn_layer = DirectedGCN(layer_size, num_labels, train_mode)
#hidden_states = torch.tensor([[0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5]])
#hidden_states = gcn_layer(hidden_states,adj, labels, adj_inv, labels_inv)
#print(hidden_states)












# import torch
# import torch.nn as nn
# from typing import Optional

# def segment_sum(embeddings, segment_ids):
#     if (embeddings.shape[0] != segment_ids.shape[0]):
#         raise ValueError('The first dimension of embeddings must equal to the segment_ids size.')

#     # This tensor will have size equal to the first tensor in our data tensor
#     tensor_return = torch.zeros_like(embeddings[:, :].clone().split(1)[0])
#     #print(tensor_return)

#     # This tensor will have size equal to the first tensor in our data tensor, remains unchanged !
#     zero_tensor = torch.zeros_like(embeddings[:, :].clone().split(1)[0])

#     first_time = True
#     sum = 0
#     index = 0
#     start = 0
#     prev_index = 0

#     segment_id = segment_ids.split(1)[0].item()
#     #print(segment_id)
    
#     #print("--- segment_ids.split(1): ")
#     #print(segment_ids.split(1))

#     for id in segment_ids.split(1):
#         #print(' ----   Iteration width id: -----')
#         #print(id.item())
#         if (id.item() == segment_id):
#             #print("equal")
            
            
#             # Ids not given in our segment_ids will give tensor with 0 value
#             for missing_id in range(prev_index, id.item()):
#                 #print("missing_id: ")
#                 #print(missing_id)
#                 if (first_time):
#                     # Initially, this tensor has 0 values in it, we must re-assign it
#                     tensor_return = zero_tensor.clone()
#                     first_time = False
#                 else:
#                     tensor_return = torch.cat((tensor_return, zero_tensor), 0)
            
#             prev_index = id.item()
        
#         else:                
#             # found diff index

        
#             # Get a list of tensors from starting row to index row from our data tensor
#             b = embeddings[start:index, :].clone()
        
#             # Create a tensor sum_t that has 0 values and shape like the first tensor
#             sum_t = torch.zeros_like(b.split(1)[0])
        
#             # Sum the value of each tensor to sum_t
#             for t in b.split(1):
#                 #print(t)
#                 sum_t += t 
        
#             #print("t: ")
#             #print(sum_t)
            
            
#             if (first_time):
#                 # Initially, this tensor has 0 values in it, we must re-assign it
#                 tensor_return = sum_t.clone()
#                 first_time = False
#             else:
#                 tensor_return = torch.cat((tensor_return, sum_t), 0)
                
                
        
#             if id.item() - prev_index > 1:
#                 # Ids not given in our segment_ids will give tensor with 0 value
#                 prev_missing_id = -1

#                 for missing_id in range(prev_index + 1, id.item()):
#                     #print("missing_id: ")
#                     #print(missing_id)
#                     if (prev_missing_id != missing_id):
#                         if (first_time):
#                             # Initially, this tensor has 0 values in it, we must re-assign it
#                             tensor_return = zero_tensor.clone()
#                             first_time = False
#                         else:
#                             tensor_return = torch.cat((tensor_return, zero_tensor), 0)
            
#             prev_index = id.item()
        
        
#             segment_id = id.item()          
        
#             start = index  
    
#         index += 1
    

    
#     # Get a list of tensors from starting row to index row from our data tensor
#     b = embeddings[start:index, :].clone()

#     # Create a tensor sum_t that has 0 values and shape like the first tensor
#     sum_t = torch.zeros_like(b.split(1)[0])
#     for t in b.split(1):
#         #print(t)
#         sum_t += t
        
#     #print("t: ")
#     #print(sum_t)
    
#     if (first_time):
#         tensor_return = sum_t.clone()
#         first_time = False
#     else:
#         tensor_return = torch.cat((tensor_return, sum_t), 0)


#     # Ids not given in our segment_ids will give tensor with 0 value
#     prev_missing_id = -1
#     for missing_id in range(prev_index, index - 1):
#         #print("missing_id: ")
#         #print(missing_id)
#         if (prev_missing_id != missing_id):
#             prev_missing_id = missing_id
#             if (first_time):
#                 # Initially, this tensor has 0 values in it, we must re-assign it
#                 tensor_return = zero_tensor.clone()
#                 first_time = False
#             else:
#                 tensor_return = torch.cat((tensor_return, zero_tensor), 0)
#     #print("    Tensor return: ")
#     #print(tensor_return)
#     return tensor_return


# def embedding_lookup_sparse(embeddings, sparse_ids, sparse_weights):
#     # print("Embedding lookup sparse")
    
#     # print("--- Embeddings:")
#     # print(embeddings.size())
    
#     # print("--- sparse_ids:")
#     # print(sparse_ids)
#     # print("sparse_ids size:")
#     # print(sparse_ids.size())
#     # print("sparse_ids values: ")
#     # print(sparse_ids._values())
#     # print("sparse_ids values size:")
#     # print(sparse_ids._values().size())
    
#     # print("--- sparse_weights:")
#     # print(sparse_weights.size())
#     # print("sparse_weights values: ")
#     # print(sparse_weights._values())
#     # print("sparse_weights values size:") 
#     # print(sparse_weights._values().size())
    
#     if (sparse_ids.size() != sparse_weights.size()):
#         raise ValueError("Shape of sparse_ids and sparse_weights are incompatible")
    
    
#     a = sparse_ids._indices()
#     # print("--- sparse_ids indices: ")
#     # print(a)    
    
#     segment_ids = sparse_ids._indices()[0] # get the tensor containing all the rows
#     # print("--- segment_ids:")
#     # print(segment_ids)
    
#     ids = sparse_ids._values()
#     # print("--- ids = sparse_ids.values(): ")
#     # print(ids)
    
#     ids, idx = torch.unique(ids, sorted=True, return_inverse=True)
#     # print("--- After unique(ids): ")
#     # print("ids: ")
#     # print(ids)
#     # print("idx: ")
#     # print(idx)
    
#     embeddings = embedding_lookup(embeddings, ids)
#     # print("--- Embeddings after embedding_lookup: ")
#     # print(embeddings)
#     # print(embeddings.size())
    
#     weights = sparse_weights._values().unsqueeze(-1)
#     # print("Weights: ")
#     # print(weights)
#     # print(weights.size())
    
    
#     origin_size = embeddings.size()
#     # print("--- origin size:")
#     # print(origin_size)
    
#     embeddings = embeddings[idx] # equivalent to tf.gather     
#     # print("--- Embeddings after embeddings[idx] ")
#     # print(embeddings)    
#     # print(embeddings.size())
    
    
#         # Reshape weights to allow broadcast
# #     ones = array_ops.fill(
# #           array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
# #     x = torch.ones(embeddings.dim() - 1)
# #     print(x.size())
# #     print(x)

# #     x = x.unsqueeze(0)
# #     print(x.size())
# #     print(x)

# #     ones = torch.ones(x.size())
# #     print("--- ones: ")
# #     print(ones.size())
# #     print(ones)
    
# #     bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],0)
# #     print("Weigths size: ")
# #     print(weights.size())
# #     bcast_weights_shape = torch.cat((torch.zeros(weights.size()), ones), 0)
    
# #     orig_weights_shape = weights.size()
# #     print("--- original weigths shape: ")
# #     print(orig_weigths_shape)
    
# #     weights = array_ops.reshape(weights, bcast_weights_shape)
    
    
# #     embeddings = math_ops.segment_sum(embeddings, segment_ids)  
#     #print("--- embeddings size: ")
#     #print(embeddings.size())
#     #print("--- weights size: ")
#     #print(weights.size())
    
# #     torch.reshape(weights, (embeddings.size()[0], -1))
# #     weights = weights.unsqueeze(-1)
# #     print("new weights size: ")
# #     print(weights.size())
    
#     embeddings *= weights
#     # print(" embeddings after multiplication: ")
#     # print(embeddings.size())
    
#     result = segment_sum(embeddings, segment_ids)
#     # print("Result: ")
#     # print(result)
#     # print(result.size())
#     embeddings = torch.zeros((sparse_ids.size()[0], origin_size[1])) # 2720 * 256
#     embeddings.new_tensor(result)

#     # print("--- Return embeddings...")
#     # print(embeddings.size())
#     return embeddings



# def sparse_dropout_mask(variable: torch.sparse.FloatTensor,
#                         keep_prob: float,
#                         train_mode: torch.Tensor) -> torch.sparse.FloatTensor:
#     """Performs dropout on a sparse tensor, depending on mode. """

#     #shape = tf.shape(variable.values)
#     shape = variable._values().size()
#     #print(shape)
#     #print(variable.values)
#     #with tf.variable_scope("dropout"):
#     if keep_prob == 1.0:
#         #return tf.fill(shape, True)
#         v = torch.empty(shape)
#         return v.fill_(True)

#     keep_prob = torch.where(train_mode, torch.tensor(keep_prob), torch.tensor(1.0))

#     #probs = tf.random_uniform(shape) + keep_prob
#     t = torch.empty(shape)
#     probs = torch.nn.init.uniform_(t) + keep_prob

#     #return tf.cast(tf.floor(probs), dtype=tf.bool)
#     return torch.floor(probs).byte()

# def sparse_retain(tensor,masked_select):
#     masked_select = torch.squeeze(masked_select)
#     c = tensor._values()
#     c = torch.squeeze(c)
#     c = c[masked_select] # remove value with dropout
#     d = tensor._indices()
#     d = d.t()
#     d = d[masked_select] # remove indices of values with dropout
#     t = torch.sparse_coo_tensor(d.t(), c,tensor.size()) # construct a new sparse tensor
#     return t


# #import torch

# #n = torch.tensor([[[1,1],[2,2],[3,3],[4,4]],[[11,11],[12,12],[13,13],[14,14]],
# #                          [[21,21],[22,22],[23,23],[24,24]]])

# def embedding_lookup(embeddings, indices):
#     #print(indices.size())
#     #print(embeddings.index_select(0, indices.view(-1)))
#     indices.cuda()
#     return embeddings.index_select(0, indices.view(-1)).view(*(indices.size() + embeddings.size()[1:]))

# #print(n)

# #ids = torch.tensor([0,2])

# #print("Embedding lookup of n: ")
# #print(embedding_lookup(n, ids))

# def sparse_fill_empty_rows(t, default_value):
#     #print(type(default_value))
#     row = t.size()[0]
#     indices = t._indices().t()
#     #print(indices.type())
#     mdict = [item[0].item() for item in indices]
#     notIn = [([i,0]) for i in range(0,row) if i not in mdict]
#     b = torch.tensor(notIn,dtype = torch.long)
#     indices = torch.cat((indices,b))
#     zeros = torch.zeros([len(notIn)],dtype = torch.long)
#     zeros.fill_(default_value)
#     if type(default_value) == float:
#         #print("float")
#         zeros = zeros.float()
#     #print(t._values().type())
#     #print(zeros.type())
#     values = torch.cat((t._values(),zeros))
#     #else:
#     #    values = torch.cat((t._values().float(),zeros))
#     ten = torch.sparse_coo_tensor(indices.t(), values,t.size())
#     return ten

# def sparse_fill_empty_rows_V2(t, default_value):
#     row = t.size()[0]
#     indices = t._indices().t()
#     values = t._values()
#     new_indices = []
#     new_values = []
#     i = 0
#     if(indices[0] is not None):
#         while(i!=indices[0][0].item()):
#             new_indices.append([i,0])
#             new_values.append(default_value)
#             i = i+1  
#     for index, item in enumerate(indices):
#         if(item[0].item()==i):
           
#             new_indices.append([item[0].item(),item[1].item()])
#             new_values.append(values[index])         
#         else:
#             i = i + 1
#             while (item[0].item() != i):
#                 new_indices.append([i,0])
#                 new_values.append(default_value)
#                 i = i+1  
#             new_indices.append([item[0].item(),item[1].item()])
#             new_values.append(values[index])          
#     i = i + 1   
#     while(i<row):
#         new_indices.append([i,0])
#         new_values.append(default_value)
#         i = i + 1  
#     new_indices_tensor = torch.tensor(new_indices)
#     ten = torch.sparse_coo_tensor(new_indices_tensor.t(), new_values,t.size())
#     return ten    
# #71-75
# #h = torch.matmul(inputs2d, self.w)
# #h = torch.sparse.mm(adj, h)
# #label_pads = sparse_fill_empty_rows(labels,0)
# #labels_weights = sparse_fill_empty_rows(adj, 0.)


# #test
# #i = torch.tensor([[0,0],[1,2],[0,1],[9,2]])
# #v = torch.tensor([1,2,3,10])
# #t = torch.sparse_coo_tensor(i.t(), v,[10,10])
# #ten = sparse_fill_empty_rows(t,0)
# #print(ten)

# #i = torch.tensor([[0,0],[1,1],[2,2],[3,3]])
# #v = torch.tensor([1,2,3,4])
# #t = torch.sparse_coo_tensor(i.t(), v,[5,5])
# #print(t)
# #b = torch.tensor(True)
# #prob = sparse_dropout_mask(t,0.5,b)
# #prob = torch.squeeze(prob)
# #print(prob)
# #tensor = sparse_retain(t,prob)
# #print(tensor)


# #c = t._values()
# #c = torch.squeeze(c)
# #c = c[prob]


# #d = t._indices()
# #d = [i for i in d if ]
# #d = d.t()
# #print(d)
# #prob = torch.squeeze(prob)
# #print(prob)
# #d = torch.masked_select(d,prob)
# #d = d[prob]
# #t = torch.sparse_coo_tensor(d.t(), c,t.size())
# #print(t)

# def element_wise_mul(sparse_matrix, dense_matrix):
#     indices = sparse_matrix._indices().t()
#     values = sparse_matrix._values()
#     for i in range(indices.size()[0]):
#         index = indices[i]
#         values[i] = values[i] * dense_matrix[0][index[1]]
#     return torch.sparse_coo_tensor(indices.t(), values,sparse_matrix.size())

# class DirectedGCN:
#     def __init__(self, layer_size: int, num_labels: int, train_mode,
#                  dropout_keep_p: float = 0.8,
#                  edge_dropout_keep_p: float=0.8,
#                  residual: Optional[bool] = True,
#                  name: str='gcn'):
#         self.layer_size = layer_size
#         self.num_labels = num_labels
#         self.train_mode = train_mode

#         self.dropout_keep_p = dropout_keep_p
#         self.edge_dropout_keep_p = edge_dropout_keep_p
#         self.residual = residual
#         self.name = name
#         self._create_weight_matrices()
#     def __call__(self, inputs, adj, labels, adj_inv, labels_inv):
#         # graph convolution, heads to dependents ("out")
#         # gates are applied through the adjacency matrix values

#         # apply sparse dropout
#         # print("Inputs: ")
#         # print(inputs)
#         # print(inputs.size())
        
#         state_dim = inputs.size()[2]
#         inputs2d = torch.reshape(inputs, [-1, state_dim])

#         to_retain = sparse_dropout_mask(
#             adj, self.edge_dropout_keep_p, self.train_mode)
#         to_retain = torch.squeeze(to_retain)

#         adj = sparse_retain(adj, to_retain).cuda()
#         labels = sparse_retain(labels, to_retain).cuda()
        
#         # apply gates        
#         gates = torch.matmul(inputs2d, self.w_gate)
    
#         #adj *= torch.t(gates)
#         adj = element_wise_mul(adj,torch.t(gates))
#         gates_bias = torch.squeeze(embedding_lookup(self.b_gate, labels._values()))
#         #print(gates_bias)
#         #print(adj._values())
#         #temp = adj._values().float() + gates_bias
#         #print(temp)
#         sigmoid = nn.Sigmoid()
#         values = sigmoid(adj._values().float() + gates_bias)

#         # dropout scaling
#         values /= torch.where(self.train_mode, torch.tensor(self.edge_dropout_keep_p), torch.tensor(1.0))

#         adj = torch.sparse_coo_tensor(adj._indices(), values, adj.size())
        
#         # graph convolution, heads to dependents ("out")
#         h = torch.matmul(inputs2d, self.w)
        
#         #h = tf.sparse_tensor_dense_matmul(adj, h)
#         h = torch.sparse.mm(adj, h).cuda()
#         #labels_pad, _ = tf.sparse_fill_empty_rows(labels, 0)
#         #labels_weights, _ = tf.sparse_fill_empty_rows(adj, 0.)
#         #print(labels)
#         #print(adj)
#         labels_pad = sparse_fill_empty_rows_V2(labels,0).cuda()
#         labels_weights = sparse_fill_empty_rows_V2(adj, 0.).cuda()
#         #print(self.b.size())
#         # print("Labels _ pad : ")
#         # print(labels_pad)
#         # print(labels_pad.size())
#         # print("Labels _ weigths : ")
#         # print(labels_weights)
#         # print(labels_weights.size())
#         b_clone = self.b.clone()
#         labels = embedding_lookup_sparse(b_clone, labels_pad, labels_weights).cuda()
        
#         h = h + labels
#         h = torch.reshape(h,inputs.size())
#         #h = tf.reshape(h, tf.shape(inputs))

#         # graph convolution, dependents to heads ("in")
#         # gates are applied through the adjacency matrix values

#         # apply sparse dropout
#         to_retain_inv = sparse_dropout_mask(
#             adj_inv, self.edge_dropout_keep_p, self.train_mode)
#         to_retain_inv = torch.squeeze(to_retain_inv)

#         adj_inv = sparse_retain(adj_inv, to_retain_inv).cuda()
#         labels_inv = sparse_retain(labels_inv, to_retain_inv).cuda()

#         # apply gates
#         gates_inv = torch.matmul(inputs2d, self.w_gate_inv)
#         #adj_inv *= torch.t(gates_inv)
#         adj_inv = element_wise_mul(adj_inv,torch.t(gates_inv))
#         gates_inv_bias = torch.squeeze(embedding_lookup(self.b_gate_inv, labels_inv._values()))
#         values_inv = sigmoid(adj_inv._values().float() + gates_inv_bias)

#         # dropout scaling
#         values_inv /= torch.where(self.train_mode, torch.tensor(self.edge_dropout_keep_p), torch.tensor(1.0)).cuda()
#         adj_inv = torch.sparse_coo_tensor(adj_inv._indices(), values_inv, adj_inv.size()).cuda()


#         # graph convolution, dependents to heads ("in")
        
#         h_inv = torch.matmul(inputs2d, self.w_inv).cuda()

#         # h_inv = tf.sparse_tensor_dense_matmul(adj_inv, h_inv)
#         h_inv = torch.sparse.mm(adj_inv, h_inv)

#         # labels_inv_pad, _ = tf.sparse_fill_empty_rows(labels_inv, 0)
#         labels_inv_pad = sparse_fill_empty_rows_V2(labels_inv,0).cuda()

#         # labels_inv_weights, _ = tf.sparse_fill_empty_rows(adj_inv, 0.)
#         labels_inv_weights = sparse_fill_empty_rows_V2(adj_inv,0).cuda()

#         b_inv_clone = self.b_inv.clone()
#         labels_inv = embedding_lookup_sparse(b_inv_clone, labels_inv_pad, labels_inv_weights,).cuda()

#         h_inv = h_inv + labels_inv
#         h_inv = torch.reshape(h_inv, inputs.size())

#         # graph convolution, loops
#         gates_loop = sigmoid(torch.matmul(inputs2d, self.w_gate_loop) + self.b_gate_loop)
#         h_loop = torch.matmul(inputs2d, self.w_loop) + self.b_loop

#         h_loop = h_loop * gates_loop
#         h_loop = torch.reshape(h_loop, inputs.size())

#         # final result is the sum of those (with residual connection to inputs)        
#         relu = nn.ReLU()
#         h = relu(h + h_inv + h_loop)

#         if self.residual:
#             print("GCN has residual connection")           
#             return h + inputs
#         else:
#             print("GCN without residual connection")
#             return h

#     def _create_weight_matrices(self):
#         """Creates all GCN weight matrices."""
#         self.w = torch.empty(self.layer_size,self.layer_size).cuda()
#         nn.init.normal_(self.w, std = 0.01)
       
#         self.w_inv = torch.empty(self.layer_size,self.layer_size).cuda()
#         nn.init.normal_(self.w_inv, std = 0.01)
        
#         self.w_loop = torch.empty(self.layer_size,self.layer_size).cuda()
#         nn.init.normal_(self.w_loop, std = 0.01)
       
#         self.w_gate = torch.empty(self.layer_size,1).cuda()
#         nn.init.normal_(self.w_gate, std = 0.01)
        
#         self.w_gate_inv = torch.empty(self.layer_size,1).cuda()
#         nn.init.normal_(self.w_gate_inv, std = 0.01)
        
#         self.w_gate_loop = torch.empty(self.layer_size,1).cuda()
#         nn.init.normal_(self.w_gate_loop, std = 0.01)
        
#         self.b_gate = torch.empty(self.num_labels).cuda()
#         nn.init.normal_(self.b_gate,mean = 0.0, std = 0.01)
        
#         self.b_gate_inv = torch.empty(self.num_labels).cuda()
#         nn.init.normal_(self.b_gate_inv,mean = 0.0, std = 0.01)
        
#         self.b_gate_loop= torch.tensor([1.]).cuda()
       
#         self.b = torch.empty(self.num_labels,self.layer_size).cuda()
#         nn.init.normal_(self.b, std = 0.01)
       
#         self.b_inv = torch.empty(self.num_labels,self.layer_size).cuda()
#         nn.init.normal_(self.b_inv, std = 0.01)
        
#         self.b_loop = torch.empty(self.layer_size).cuda()
#         nn.init.normal_(self.b_loop, std = 0.01)
        
#layer_size = 5
#num_labels = 3
#train_mode = torch.tensor(True)
#i = torch.tensor([[0,0],[0,1],[1,1],[2,2],[2,4],[3,3],[4,3],[4,4]])
#v = torch.tensor([1,1,1,1,1,1,1,1])
#adj = torch.sparse_coo_tensor(i.t(), v,[5,5])
#labels = torch.sparse_coo_tensor(i.t(), v,[5,5])
#i = torch.tensor([[0,0],[1,0],[1,1],[2,2],[3,3],[3,4],[4,4],[4,2]])
#adj_inv = torch.sparse_coo_tensor(i.t(), v,[5,5])
#labels_inv = torch.sparse_coo_tensor(i.t(), v,[5,5])
#gcn_layer = DirectedGCN(layer_size, num_labels, train_mode)
#hidden_states = torch.tensor([[0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5],
#                            [0.1,0.2,0.3,0.4,0.5]])
#hidden_states = gcn_layer(hidden_states,adj, labels, adj_inv, labels_inv)
#print(hidden_states)
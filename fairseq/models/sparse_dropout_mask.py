import torch


def sparse_dropout_mask(variable: torch.sparse.FloatTensor,
                        keep_prob: float,
                        train_mode: torch.Tensor) -> torch.sparse.FloatTensor:
    """Performs dropout on a sparse tensor, depending on mode. """

    #shape = tf.shape(variable.values)
    shape = variable.values.size()


    #with tf.variable_scope("dropout"):
    if keep_prob == 1.0:
        #return tf.fill(shape, True)
        v = torch.empty(shape)
        return v.fill_(True)

    keep_prob = torch.where(train_mode, keep_prob, 1.0)

    #probs = tf.random_uniform(shape) + keep_prob
    t = torch.empty(shape)
    probs = torch.nn.init.uniform_(t) + keep_prob

    #return tf.cast(tf.floor(probs), dtype=tf.bool)
    return torch.floor(probs).byte()


def embedding_lookup(embeddings, indices):
     return embeddings.index_select(0, indices.view(-1)).view(*(indices.size() + embeddings.size()[1:]))

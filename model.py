import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore import ops as P
import mindspore.common.initializer as init
import config

class Net(nn.Cell):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )
        self.cat = P.Concat(axis=1)
        self.norm = nn.Norm(axis=1, keep_dims=True)

    def construct(self, v, q, q_len):
        q = self.text(q, q_len)

        v = v / (self.norm(v).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)
        combined = self.cat((v, q))
        answer = self.classifier(combined)
        return answer


class TextProcessor(nn.Cell):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(keep_prob=1 - drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1,
                            batch_first=True)
        self.features = lstm_features
        self.squeeze = P.Squeeze(axis=0)
    def construct(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))

        h0 = Tensor(np.ones([1, q.shape[0], self.features]).astype(np.float32))
        c0 = Tensor(np.ones([1, q.shape[0], self.features]).astype(np.float32))
        
        _, (_, c) = self.lstm(tanhed, (h0, c0))

        # return c.squeeze(0) # only supported from 1.2.x
        return self.squeeze(c)


class Attention(nn.Cell):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, has_bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Dense(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(1 - drop)
        self.relu = nn.ReLU()

    def construct(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x


class Classifier(nn.SequentialCell):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.insert_child_to_cell('drop1', nn.Dropout(keep_prob=1 - drop))
        self.insert_child_to_cell('lin1', nn.Dense(in_features, mid_features))
        self.insert_child_to_cell('relu', nn.ReLU())
        self.insert_child_to_cell('drop2', nn.Dropout(keep_prob=1 - drop))
        self.insert_child_to_cell('lin2', nn.Dense(mid_features, out_features))
        self.cell_list = list(self._cells.values())


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    softmax = P.Softmax(axis=-1)
    unsqueeze = P.ExpandDims()
    reduce_sum = P.ReduceSum()

    n, c = input.shape[:2]
    glimpses = attention.shape[1]

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = unsqueeze(softmax(attention), 2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = reduce_sum(weighted, -1) # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.shape
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled
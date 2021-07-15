import sys
import os.path
import mindspore
from mindspore import Tensor, nn, Model, context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import ops as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.parameter import ParameterTuple
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.nn.loss.loss import _Loss
import numpy as np
from tqdm import tqdm
import config
import data
import model
import utils
import mindspore.context as context
import json
import math

class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell be trained without inputs.
    Backward graph will be created in the construct function to do parameter updating. Different
    parallel modes are available to run the training.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True)
        self.sens = sens

    def construct(self, v, q, a, item, q_len):
        weights = self.weights
        loss = self.network(v, q, a, item, q_len)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(v, q, a, item, q_len)
        return F.depend(loss, self.optimizer(grads))

class NLLLoss(_Loss):
    '''
       NLLLoss function
    '''
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.reduce_sum = P.ReduceSum()
        self.log_softmax = P.LogSoftmax(axis=0)

    def construct(self, logits, label):
        nll = -self.log_softmax(logits)
        loss = self.reduce_sum(nll * label / 10, axis=1).mean()
        return self.get_loss(loss)

class OutLossAccuracyWrapper(nn.Cell):
    """
    The highest level cell for evaluation, wrapped with NLL Loss and accuracy. (use it directly)
    
    Output:
        output: a Tensor of shape (batch_size, config.max_answers) (logits)
        loss: a scalar value
        accuracy: a Tensor of shape (batch_size, 1)
    """
    def __init__(self, backbone):
        super(OutLossAccuracyWrapper, self).__init__()
        self.net = backbone
        self._loss_fn = NLLLoss()

    def construct(self, v, q, a, item, q_len):
        output = self.net(v, q, q_len)
        loss = self._loss_fn(output, a)
        accuracy = utils.batch_accuracy(output, a)
        return output, loss, accuracy

class WithLossCell(nn.Cell):
    """
    The cell wrapped with NLL loss, for train only
    """
    def __init__(self, backbone):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._loss_fn = NLLLoss()
        self._backbone = backbone
        self.reduce_sum = P.ReduceSum()

    def construct(self, v, q, a, item, q_len):
        out = self._backbone(v, q, q_len)
        loss = self._loss_fn(out, a)
        return loss

class TrainNetWrapper(nn.Cell):
    """
    The highest level train cell. (use it directly)
    """
    def __init__(self, backbone):
        super(TrainNetWrapper, self).__init__(auto_prefix=False)
        self.net = backbone
        
        loss_net = WithLossCell(backbone)
        optimizer = nn.Adam(params=net.trainable_params(), learning_rate=config.initial_lr)
        
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)

    def construct(self, v, q, a, item, q_len):
        loss = self.loss_train_net(v, q, a, item, q_len)
        output = self.net(v, q, q_len)
        accuracy = utils.batch_accuracy(output, a)
        # print(accuracy)
        return loss, accuracy

def run(net, loader, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    arg_max = P.Argmax(axis=1, output_type=mindspore.int32)
    cat = P.Concat(axis=0) # Warning: `Concat` a list of tensors is not supported in mindspore 1.1.x

    if train:
        net.set_train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.set_train(False)
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0, total=math.ceil(len(loader.source) / config.batch_size))
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    for v, q, a, idx, q_len in tq:
        if train:
            loss, acc = net(v, q, a, idx, q_len)
        else:
            output, loss, acc = net(v, q, a, idx, q_len)
            answer = arg_max(output)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1))
        
        # Update loss and accuracy in console line
        loss_tracker.append(loss.asnumpy())
        for a in acc.asnumpy():
            acc_tracker.append(a)
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
    
    if not train:
        # Cast to python types for JSON serialization
        answ = list(map(int, list(cat(answ).asnumpy())))
        accs = list(cat(accs).asnumpy().astype(float))
        idxs = list(map(int, list(cat(idxs).asnumpy())))
        return answ, accs, idxs

if __name__ == '__main__':
    if config.device == 'GPU': os.environ['CUDA_VISIBLE_DEVICES'] = '1' # select GPU if necessary
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device)

    from datetime import datetime
    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.ckpt'.format(name))
    print('will save to {}'.format(target_name))
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = model.Net(train_loader.source.num_tokens)
    if config.pretrained:
        param_dict = load_checkpoint(config.pretrained_model_path)
        if param_dict is not None: print("Successfully loaded pretrained model from {}.".format(config.pretrained_model_path))
        load_param_into_net(net, param_dict)

    tracker = utils.Tracker()
    train_net = TrainNetWrapper(net) # for train
    eval_net = OutLossAccuracyWrapper(net) # for evaluation
    step = 0

    for epoch in range(config.epochs):
        # train_loader = data.get_loader(train=True) # not sure if it matters?
        
        """
        Wrapped train with `tqdm`
        """
        run(train_net, train_loader, tracker, train=True, prefix='train', epoch=epoch)
        r = run(eval_net, val_loader, tracker, train=False, prefix='val', epoch=epoch)
        
        # Calculate the validate accuracy mean of each batch
        val_acc = []
        for acc_list in tracker.to_dict()['val_acc']:
            val_acc.append(np.mean(acc_list).astype(float))

        results = {
            'name': name,
            # 'tracker': tracker.to_dict(),
            'accuracy': val_acc,
            'config': config_as_dict,
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'vocab': train_loader.source.vocab,
        }

        # Save model as CKPT every 5 epochs
        if epoch % 5 == 0: mindspore.save_checkpoint(train_net.net, ckpt_file_name=os.path.join('logs', '{}.ckpt'.format(name)))
        
        # Save train meta info as JSON
        with open(os.path.join('logs', 'TrainRecord_{}.json'.format(name)), 'w') as fp:
            fp.write(json.dumps(results))
        
        # train_loader.reset() # not sure if it matters?

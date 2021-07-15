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
    def __init__(self, backbone):
        super(TrainNetWrapper, self).__init__(auto_prefix=False)
        self.net = backbone
        
        loss_net = WithLossCell(backbone)
        optimizer = nn.Adam(params=net.trainable_params(), learning_rate=config.initial_lr)
        
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)

    def construct(self, v, q, a, item, q_len):
        loss = self.loss_train_net(v, q, a, item, q_len)
        accuracy = Tensor(0.35)
        return loss, accuracy

def run(net, loader, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.set_train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.set_train(False)
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    for v, q, a, idx, q_len in tq:
        if train:
            loss, acc = net(v, q, a, idx, q_len)
        else:
            print("Evaluating...")
        
        loss_tracker.append(loss.asnumpy())
        acc_tracker.append(acc.asnumpy())
        # acc_tracker.append(acc.mean())
        # for a in acc:
        #     acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

if __name__ == '__main__':
    if config.device == 'GPU': os.environ['CUDA_VISIBLE_DEVICES'] = '1' # select GPU if necessary
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device)

    from datetime import datetime
    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.ckpt'.format(name))
    print('will save to {}'.format(target_name))
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    train_loader = data.get_loader(train=True)
    # val_loader = data.get_loader(val=True)

    net = model.Net(train_loader.source.num_tokens)
    if config.pretrained:
        param_dict = load_checkpoint(config.pretrained_model_path)
        if param_dict is not None: print("Successfully loaded pretrained model from {}.".format(config.pretrained_model_path))
        load_param_into_net(net, param_dict)

    tracker = utils.Tracker()
    train_net = TrainNetWrapper(net)
    step = 0

    for epoch in range(config.epochs):
        # train_loader = data.get_loader(train=True) # not sure if it matters?

        """
        Hand-crafted train wiht `for` loop
        """
        # train_net.set_train()
        # for v, q, a, idx, q_len in train_loader:
        #     train_result = train_net(v, q, a, idx, q_len)
        #     train_loss = train_result[0]
        #     train_acc = train_result[1]
        #     print("T{} step {}: loss = {}, acc = {}".format(epoch, step, train_loss, train_acc))
        #     step += 1
        
        """
        Wrapped train with `tqdm`
        """
        run(train_net, train_loader, tracker, train=True, prefix='train', epoch=epoch)

        # train_loader.reset() # not sure if it matters??
        # break

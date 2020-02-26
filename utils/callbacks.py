from keras.callbacks import Callback
import keras.layers as layers
import keras.backend as K

class Divide_lr(Callback):
    def __init__(self, interval, divide_factor):
        assert divide_factor>0, 'factor should be greater than 1'
        assert interval>1, 'interval should be greater than 1'
        self.inverval = interval
        self.factor = divide_factor
    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.inverval == 0 and epoch > 1:
            learning_rate = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr,  learning_rate/self.factor)
            print('Divide learning rate by', self.factor, 'Current learning rate',learning_rate/self.factor)
    def get_config(self):
        config = {'class': type(self).__name__,
          'interval': self.inverval,
          'divide_factor': self.factor}
        return config

class Step(Callback):
    def __init__(self, steps, learning_rates, verbose = 0):
        assert len(steps) == (len(learning_rates)-1), 'lengths of steps and learning_rates not match'
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose
    def change_lr(self, new_lr):
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)
    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])
    def get_config(self):
        config = {'class': type(self).__name__,
          'steps': self.steps,
          'learning_rates': self.lr,
          'verbose': self.verbose}
        return config

class BNDecayScheduler(Callback):
    def __init__(self, bn_init, decay_rate, interval, clip = 0.999):
        self.bn_init = bn_init
        self.decay_rate = decay_rate
        self.interval = interval
        self.clip = clip
    def on_epoch_begin(self, epoch,logs={}):
        if epoch == 0:
            self.init_bn_decay()
        elif epoch%self.interval == 0:
            self.update_bn_decay()
    def update_bn_decay(self):
        for layer in self.model.layers:
            if isinstance(layer, layers.BatchNormalization):
                temp = 1 - layer.momentum
                layer.momentum = min(1 - temp*self.decay_rate, self.clip)
                temp = layer.momentum
        print('Update BN momentum to',temp)
    def init_bn_decay(self):
        for layer in self.model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.momentum = self.bn_init
        print('Initilize BN momentum to', self.bn_init)
    def get_config(self):
        config = {'class': type(self).__name__,
                    'bn_init':self.bn_init,
                    'decay_rate':self.decay_rate,
                    'interval':self.interval,
                    'clip':self.clip}
        return config
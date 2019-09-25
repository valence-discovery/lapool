import os
import sys
import numpy as np
import torch
import torch.optim as optim
import traceback
import torch.optim as optim
from poutyne.framework.callbacks.lr_scheduler import _PyTorchLRSchedulerWrapper
from gnnpooling.utils.graph_utils import data2mol, convert_mol_to_smiles
from poutyne.framework.optimizers import all_optimizers_dict
from poutyne.framework.callbacks import *
from poutyne.framework.iterators import EpochIterator, StepIterator, _get_step_iterator
from torch.utils.data import DataLoader, Dataset, TensorDataset
from poutyne.framework import Model
from poutyne import torch_to_numpy, numpy_to_torch
from poutyne.framework.warning_manager import warning_settings
from tensorboardX import SummaryWriter

name2callback = {"early_stopping": EarlyStopping,
                 "reduce_lr": ReduceLROnPlateau,
                 "norm_clip": ClipNorm}

def params_getter(param_dict, prefix):
    r"""
    Filter a parameter dict to keep only a list of relevant parameters
    """
    # Check the instances of the prefix
    if isinstance(prefix, str):
        prefix = [prefix]
    if not (isinstance(prefix, list) or isinstance(prefix, tuple)):
        raise ValueError(
            "Expect a string, a tuple of strings or a list of strings, got {}".format(type(prefix)))
    if not all([isinstance(this_prefix, str) for this_prefix in prefix]):
        raise ValueError("All the prefix must be strings")

    # Create the new dictionary
    new_dict = {}
    for pkey, pval in param_dict.items():
        for this_prefix in prefix:
            if pkey.startswith(this_prefix):
                new_dict[pkey.split(this_prefix, 1)[-1]] = pval

    return new_dict

class ExtCallbackList(CallbackList):
    r"""Extending rhe default CallBackList class to allow additional methods"""
    
    def on_backward_start(self, batch, loss=None):
        """
        This method is called after the loss computation but before the backpropagation and optimization step
        
        Arguments
        ---------
            batch: int
                The batch number.
            loss: `torch.FloatTensor`

        """
        for callback in self.callbacks:
            backward_fn = getattr(callback, 'on_backward_start', None)
            if callable(backward_fn):
                backward_fn(batch, loss)

    def __repr__(self):
        return "-> ".join(map(str, self.callbacks))

class TrainerCheckpoint(ModelCheckpoint):
    r"""
    Save and load trainer so it can be reloaded after every epoch. See Also
    `poutyne.framework.PeriodicSaveCallback` for the arguments' descriptions.

    Arguments
    ----------
        restore_best: bool
            If `restore_best` is true, the weights of the
            network will be reset to the last best checkpoint done, at the end of training.
            This option only works when `save_best_only` is also true.
            (Default value = False)

    See Also
    --------
        poutyne.framework.PeriodicSaveCallback
    """

    def save_file(self, fd, epoch, logs):
        print("==> Saving model state for epoch {}".format(epoch))
        # also send an event here to notify the last time we have saved something 
        self.model.snapshot(fd, epoch)
        return fd

    def on_train_end(self, logs):
        if self.restore_best:
            if self.best_filename is not None:
                if self.verbose:
                    print('==> Restoring model from %s' % self.best_filename)
                self.model.restore(self.best_filename)
            else:
                warnings.warn('No weights to restore!')

class Trainer(Model):
    r"""
    Trainer class for running deep learning models. If you want to provide
    specific parameters for the optimizer, use keywords that start by `optimizer__`,
    `optim__` or `op__`, the rest of the keywords should correspond to a parameter
    of the optimizer.

    Arguments
    ----------
        net: nn.Module
            Neural network model that needs to be trained
        loss_fn: callable
            A callable that takes at least two arguments
            corresponding to the predicted values and the ground truth
        metrics: dict, optional
            A dict of key, value, where each key
            corresponds to a metric name and the value is the metric function.
            Although there isn't any default, this parameter usually needs to be 
            provided, if you want to evaluate your model.
            (Default value = {})
        optimizer: optim.Optimizer or str, optional
            The optimizer to use. If a
            string is provided, it should be one of ['adam', 'sgd']
            (Default value ='adam' for the adam optimizer)
        gpu: bool, optional
            Whether to run experiments on the GPU using cuda.
            (Default value = False)
        tasks: list, optional
            The list of tasks if in multitask mode
            (Default value = [])
        model_dir: str, optional
            A path to a folder where all logs and model
            files should be saved during training. Use the default empty string
            to save in the current folder.
            (Default value = "")
        snapshot_path: str, optional
            A path to a previous checkpoint from which the training should be resume training.
            (Default value = "")
        **kwargs: Arbitrary keyword arguments. 
            - To provide parameters for the optimizer, preceed all keywords by `optim__`, `op__` or `optimizer__`

    """

    def __init__(self, net, loss_fn, metrics={}, optimizer='adam', gpu=False, tasks=[], model_dir="", snapshot_path=None, **kwargs):
        self.model = net
        self.loss_function = loss_fn
        # setting the optimizer
        if isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer
        else:
            if isinstance(optimizer, str):
                optimizer = all_optimizers_dict[optimizer.lower()]
            if not issubclass(optimizer, optim.Optimizer):
                raise ValueError("Optimizer could not be processed")
            # Permissive much ^_^
            opt_params = params_getter(
                kwargs, ("optim__", "op__", "optimizer__"))
            self.optimizer = optimizer(self.model.parameters(), **opt_params)

        self.gpu = gpu
        # Automatically send job to GPU if init asked for gpu
        self.tasks = tasks
        if metrics != {}:
            self.metrics_names, self.metrics = [
                list(x) for x in zip(*metrics.items())]
        else:
            self.metrics_names = []
            self.metrics = []
        self.model_dir = model_dir
        self.is_trained = False
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        self.device = None
        self.writer = None
        self.initial_epoch = 0
        if snapshot_path:
            self.initial_epoch = self.restore(snapshot_path)

    def network_state_dict(self):
        return self.model.state_dict()

    def optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def cuda(self, *args, **kwargs):
        if self.gpu and torch.cuda.is_available():
            super().cuda(*args, **kwargs)

    def _dataloader_from_dataset(self, dt, batch_size=None, shuffle=False):
        assert batch_size is not None, \
            "batch_size should not be None. Please, report this as a bug."
        assert isinstance(dt, Dataset), \
            "Expect a Dataset, but got {}".format(type(dt))
        generator = DataLoader(dt, batch_size, shuffle=shuffle)
        return generator

    def fit(self, train_dt, valid_dt, epochs=1000, shuffle=False, batch_size=32,
            callbacks=[], log_path=None, checkpoint=None, tboardX=".logs", **kwargs):
        r"""
        Trains the model on a dataset. This method creates generators and calls
        the ``fit_generator`` method. Note that all additional parameters of poutyne.framework.Model.fit
        such as `steps_per_epoch`, `initial_epoch`, `verbose`  and `validation_steps` remain valid.
        You can provide a generator_fn parameter if examples need to be generated in a specific way
        for your dataset. Otherwise the default pytorch dataloader will be used.
        To enabble early stopping and LR reduction, the following keyboard can be provided as a shortcut:
        "early_stopping", "reduce_lr" . They can either be set to True, or corresponds to dict of arguments
        for the corresponding callbacks. Additionally, you can also set "norm_clip" (for gradient norm cliping)
        to the maximum value you want.

        Arguments
        ----------
            train_dt: Union[Tensor, np.ndarray] or Dataset
                Training dataset.
            valid_dt: Union[Tensor, np.ndarray] or Dataset
                Validation dataset.
            epochs: int, optional
                Number of times the entire training dataset is seen.
                (Default value = 1000)
            shuffle (bool): Whether to shuffle the generator
            batch_size: int, optional
                Number of samples given to the network at one time.
                (Default value = 32)
            callbacks: list of poutyne.framework.Callback, optional
                List of callbacks
                that will be called during training. (Default value = [])
            log_path: str, optional
                Name or path of a file to log the training result per epoch
                (Default value is None)
            checkpoint: str or boolean, optional
                Path to a file for saving the current models 
                during training. This is useful if you want to reload current states to resume training. 
                If a file is not Provided, a checkpoint will be saved under the following format : 
                'model.epoch:{cur_epoch}-loss:{val_loss}.pth.tar'. Note that you could provide argument 
                for the checkpoint callback object, using the `checkpoint__`, check__`, `c__` keywords.
            tboardX: str or None, optional, default=None
                Name or path of a tensorboard directory used to save the training results.
            **kwargs: various named parameters

        Returns
        --------
            List of dict containing the history of each epoch.

        """
        valid_generator = None
        generator_fn = kwargs.get(
            "generator_fn", self._dataloader_from_dataset)
        if isinstance(train_dt, Dataset):
            train_generator = generator_fn(
                train_dt, batch_size=batch_size, shuffle=shuffle)
        else:
            train_generator = self._dataloader_from_data(
                *train_dt, batch_size=batch_size)

        if valid_dt:
            if isinstance(valid_dt, Dataset):
                valid_generator = generator_fn(
                    valid_dt, batch_size=batch_size, shuffle=shuffle)
            else:
                valid_generator = self._dataloader_from_data(
                    valid_dt, batch_size=batch_size)

        callbacks = (callbacks or [])
        # Check if early stopping is asked for
        for name in ["early_stopping", "reduce_lr"]:
            cback = kwargs.get(name)
            if cback:
                if not isinstance(cback, dict):
                    cback = {}
                try:
                    cback = name2callback[name](**cback)
                    callbacks += [cback]
                except:
                    print("Exception in callback {}".format(name))
                    traceback.print_exc(file=sys.stdout)

        clip_val = kwargs.get("norm_clip", None)
        if clip_val is not None:
            clip_cback = name2callback["norm_clip"](self.model.parameters(), clip_val)
            callbacks.append(clip_cback)
            
        if checkpoint:
            if isinstance(checkpoint, bool):
                checkpoint = 'model.epoch:{epoch:02d}-loss:{val_loss:.2f}.pth.tar'
            checkpoint = os.path.join(self.model_dir, checkpoint)
            checkpoint_params = dict(monitor='val_loss', save_best_only=True,
                                     temporary_filename=checkpoint + ".tmp")
            checkpoint_params.update(params_getter(
                kwargs, ("checkpoint__", "check__", "c__")))
            check_callback = TrainerCheckpoint(checkpoint, **checkpoint_params)
            callbacks += [check_callback]
        
        if log_path:
            log_path = os.path.join(self.model_dir, log_path)
            logger = CSVLogger(
                log_path, batch_granularity=False, separator='\t')
            callbacks += [logger]

        if tboardX:
            if isinstance(tboardX, dict):
                tboardX["logdir"] = os.path.join(self.model_dir, tboardX["logdir"])
            else:
                tboardX = {"logdir" : os.path.join(self.model_dir, tboardX)}
            try:
                self.writer = SummaryWriter(**tboardX) # this has a purpose, which is to access the writer from the GradFlow Callback
            except:
                res = tboardX.pop("logdir")
                tboardX["log_dir"] = res
                self.writer = SummaryWriter(**tboardX) 
            callbacks += [TensorBoardLogger(self.writer)]

        
        return self.fit_generator(train_generator,
                                  valid_generator=valid_generator,
                                  epochs=epochs,
                                  steps_per_epoch=kwargs.get(
                                      "steps_per_epoch"),
                                  validation_steps=kwargs.get(
                                      "validation_steps"),
                                  initial_epoch=kwargs.get("initial_epoch", 1),
                                  verbose=kwargs.get("verbose", True),
                                  callbacks=callbacks)

    def fit_generator(self, train_generator, valid_generator=None, *,
                      epochs=1000, steps_per_epoch=None, validation_steps=None,
                      initial_epoch=1, verbose=True, callbacks=[]):
        # pylint: disable=too-many-locals
        """
        Trains the model on a dataset using a generator.

        Arguments
        ----------
            train_generator: Generator-like object for the training dataset.
                The generator must yield a tuple ``(x, y)`` where ``x`` is a
                batch of the training dataset and ``y`` is the corresponding
                ground truths. ``y`` should be a Tensor or a Numpy array with
                the first dimension being the batch size since ``len(y)`` is
                taken as the batch size. The loss and the metrics are averaged
                using this batch size. If ``y`` is not a Tensor or a Numpy
                array, then a warning is raised and the "batch size" defaults
                to 1.

                If the generator does not have a method ``__len__()``, either
                the ``steps_per_epoch`` argument must be provided, or the
                iterator returned raises a StopIteration exception at the end
                of the training dataset. PyTorch DataLoaders object do provide a
                ``__len__()`` method.

                Before each epoch, the method ``__iter__()`` on the generator is
                called and the method ``__next__()`` is called for each step on
                resulting object returned by ``__iter__()``. Notice that a call
                to ``__iter__()`` on a generator made using the python keyword
                ``yield`` returns the generator itself.
            valid_generator: optional
                Generator-like object for the
                validation dataset. This generator is optional. The generator is
                used the same way as the  generator ``train_generator``. If the
                generator does not have a method ``__len__()``, either the
                ``validation_steps`` or the ``steps_per_epoch`` argument must be
                provided or the iterator returned raises a StopIteration
                exception at the end of the validation dataset.
                (Default value = None)
            epochs: int
                Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch: int, optional
                Number of batch used during one
                epoch. Obviously, using this argument may cause one epoch not to
                see the entire training dataset or see it multiple times.
                (Defaults the number of steps needed to see the entire
                training dataset)
            validation_steps: int, optional
                Same as for ``steps_per_epoch``
                but for the validation dataset. (Defaults to ``steps_per_epoch``
                if provided or the number of steps needed to see the entire
                validation dataset)
            initial_epoch: int, optional
                Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose: bool
                Whether to display the progress of the training.
                (Default value = True)
            callbacks: list of poutyne.framework.Callback
                List of callbacks that will be called during training. 
                (Default value = [])

        Returns
        -------
            List of dict containing the history of each epoch.

        Examples
        --------

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function)
                history = model.fit_generator(train_generator,
                                              valid_generator,
                                              epochs=num_epochs,
                                              verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 0.4048105351626873, 'val_loss': 0.35831213593482969}
                {'epoch': 2, 'loss': 0.27947457544505594, 'val_loss': 0.25963697880506514}
                {'epoch': 3, 'loss': 0.20913131050765515, 'val_loss': 0.20263003259897233}
                ...

        """
        initial_epoch = self.initial_epoch or initial_epoch # use model epoch number if defined
        self._transfer_optimizer_state_to_right_device()

        if verbose:
            callbacks = [ProgressionCallback()] + callbacks
        callback_list = ExtCallbackList(callbacks)
        callback_list.set_model(self)

        self.stop_training = False
        epoch_iterator = EpochIterator(train_generator, valid_generator,
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_steps=validation_steps,
                                       initial_epoch=initial_epoch,
                                       callback=callback_list,
                                       metrics_names=self.metrics_names)

        for train_step_iterator, valid_step_iterator in epoch_iterator:
            self.model.train(True)
            with torch.enable_grad():
                for step, (x, *y) in train_step_iterator:
                    step.loss, step.metrics, _ = self._fit_batch(x, *y,
                                                                 callback=callback_list,
                                                                 step=step.number)
                    step.size = self._get_batch_size(x, *y)

            if valid_step_iterator is not None:
                self._validate(valid_step_iterator)

            epoch_iterator.stop_training = self.stop_training

        self.is_trained = True
        return epoch_iterator.epoch_logs

    def _fit_batch(self, x, *y, callback=Callback(), step=None, return_pred=False):
        r"""Fit on batch"""
        self.optimizer.zero_grad()

        loss_tensor, metrics, pred_y = self._compute_loss_and_metrics(
            x, y, return_loss_tensor=True, return_pred=return_pred
        )

        callback.on_backward_start(step, loss_tensor)
        loss_tensor.backward()
        callback.on_backward_end(step)
        self.optimizer.step()

        loss = float(loss_tensor)
        return loss, metrics, pred_y

    def predict(self, x):
        r"""
        Returns the predictions of the network given an input ``x``, where the output tensors
        are converted into Numpy arrays.

        Arguments
        ---------
            x : Union[torch.Tensor, np.ndarray]
                Input Dataset (should not contains output) for which the model should predict readout.

        Returns
        -------
            Numpy arrays of the predictions.
        """
        pred_y = []
        self.model.eval()
        with torch.no_grad():
            x = self._process_input(x)
            pred_y.append(torch_to_numpy(self.model(x)))
        return np.concatenate(pred_y)

    def test_generator(self, generator, *, steps=None):
        """
        Returns the predictions of the network given batches of samples ``x``,
        where the tensors are converted into Numpy arrays.

        Arguments
        ---------
            generator: Generator-like object for the dataset. 
                The generator must yield a batch of samples. See the ``fit_generator()`` method for
                details on the types of generators supported. These method was added to allow using the same generator
                for both train, valid and test. Here, we expect the same generator to return both x and y then.
            steps (int, optional): Number of iterations done on
                ``generator``. (Defaults the number of steps needed to see the
                entire dataset)

        Returns:
            List of the predictions of each batch with tensors converted into
            Numpy arrays.
        """
        if steps is None and hasattr(generator, '__len__'):
            steps = len(generator)
        pred_y = []
        self.model.eval()
        with torch.no_grad():
            for _, (x, *y) in _get_step_iterator(steps, generator):
                x, *y = self._process_input(x, *y)
                pred_y.append(torch_to_numpy(self.model(x)))
        return np.concatenate(pred_y)

    def evaluate(self, x, *y, batch_size=32, return_pred=False, **kwargs):
        r"""
        Computes the loss and the metrics of the network on batches of samples
        and optionaly returns the predictions.

        Arguments
        ----------
            x: Union[Tensor, np.ndarray
                Dataset.
            y: Union[Tensor, np.ndarray
                Dataset ground truths.
            batch_size: int
                Number of samples given to the network at one
                time. (Default value = 32)
            return_pred: bool, optional
                Whether to return the predictions for
                ``x``. (Default value = False)

        Returns
        -------
            Float ``loss`` if no metrics were specified and ``return_pred`` is
            false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is a Numpy array of the predictions.
        """

        generator_fn= kwargs.get(
            "generator_fn", self._dataloader_from_dataset)
        generator = generator_fn(x, *y, batch_size=batch_size)
        steps = kwargs.get('steps', None)
        if steps is None:
            try:
                steps = len(generator)
            except Exception as e:
                print('Must specify steps if defining generator.')
                raise e

        ret = self.evaluate_generator(
            generator, steps=steps, return_pred=return_pred)
        if return_pred:
            ret = (*ret[:-1], np.concatenate(ret[-1]))
        return ret

    def evaluate_on_batch(self, x, *y, return_pred=False):
        r"""
        Computes the loss and the metrics of the network on a single batch of
        samples and optionaly returns the predictions.

        Arguments
        ----------
            x: Union[Tensor, np.ndarray
                Batch.
            y: Union[Tensor, np.ndarray
                Batch ground truths.
            return_pred: bool, optional
                Whether to return the predictions for
                ``x``. (Default value = False)

        Returns
        -------
            Float ``loss`` if no metrics were specified and ``return_pred`` is
            false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the predictions with tensors converted into Numpy
            arrays.
        """
        self.model.eval()
        with torch.no_grad():
            loss, metrics, pred_y = self._compute_loss_and_metrics(
                x, y, return_pred=return_pred)
        return self._format_return(loss, metrics, pred_y, return_pred)

    def _validate(self, step_iterator, return_pred=False):
        pred_list = None
        if return_pred:
            pred_list = []
        self.model.eval()
        with torch.no_grad():
            for step, (x, *y) in step_iterator:
                step.loss, step.metrics, pred_y = self._compute_loss_and_metrics(
                    x, y, return_pred=return_pred
                )
                if return_pred:
                    pred_list.append(pred_y)

                step.size = self._get_batch_size(x, *y)
        return step_iterator.loss, step_iterator.metrics, pred_list

    def _compute_loss_and_metrics(self, x, y, *, return_loss_tensor=False, return_pred=False):
        x, *y = self._process_input(x, *y)
        pred_y = self.model(x)   
        loss = self.loss_function(pred_y, *y)
        if not return_loss_tensor:
            loss = float(loss)
        with torch.no_grad():
            metrics = self._compute_metrics(pred_y, y)

        pred_y = torch_to_numpy(pred_y) if return_pred else None
        return loss, metrics, pred_y

    def _compute_metrics(self, pred_y, y):
        if isinstance(pred_y, tuple):
            pred_y = pred_y[0]
            
        return np.array([float(metric(pred_y.detach(), *y)) for metric in self.metrics])

    def _get_batch_size(self, *args):
        for val in args:
            if torch.is_tensor(val) or isinstance(val, list):
                return len(val)
        if warning_settings['batch_size'] == 'warn':
            warnings.warn("When 'x' or 'y' are not tensors nor Numpy arrays, "
                          "the batch size is set to 1 and, thus, the computed "
                          "loss and metrics at the end of each epoch is the "
                          "mean of the batches' losses and metrics. To disable "
                          "this warning, set\n"
                          "from poutyne.framework import import warning_settings\n"
                          "warning_settings['batch_size'] = 'ignore'")
        return 1



    def save(self, path, model=None):
        r"""
        Saves the current model to specified path, to it can be loaded for inference.

        Arguments
        ----------
            model: model to save
            path: str
                path to the file where the model willl be saved
        """
        if not model:
            model = self.model
        torch.save(model, path)
        return path


    def snapshot(self, path, epoch=1, save_loss=False):
        r"""
        Create a snapshot of the current training state, so it can be resumed later.

        Arguments
        ----------
            path: str
                path to the file where the model and optimizer states will be saved.
                Note that they need to support serialization.
            epoch: int, optional
                Current epoch number.
                (Default value =1)
            save_loss: bool, optional
                Whether to save the loss function. This option
                require the loss function to be serializable by Python's pickle utility
        """
        state = {
            'epoch': epoch,
            'net': self.network_state_dict(),
            'optimizer': self.optimizer_state_dict(),
            'tasks': self.tasks,
            'model_dir': self.model_dir,
            'gpu': self.gpu,
        }
        if save_loss: 
            state.update(loss=self.loss_function)
        torch.save(state, path)


    def resume_fit(self, model_path, generator=False, **kwargs):
        r"""
        Resume training from saved model

        Arguments
        ----------
            model_path: str
                path to the file where the current training states are saved
            generator: bool, optional
                Whether to fit using a `fit_generator` or `fit`
                (Default value = False)
            **kwargs: named parameters for fitting function (should match either fit 
                or fit_generator) depending on the value of generator

        Returns
        -------
            The results of fitting the model

        """
        epoch = self.restore(model_path)
        kwargs["initial_epoch"] = epoch
        if generator:
            return self.fit_generator(**kwargs)
        else:
            return self.fit(**kwargs)

    def restore(self, path):
        r"""
        Restore the training state (model, optimizer) saved in path, inside the current
        instance of the class, then return the next epoch number. If a loss function is
        not currently defined, it is loaded from the pickle object.

        Arguments
        ----------
            path: str
                path to the file containing the serialized model.

        Returns
        -------
            The last epoch saved + 1 
        """
        epoch = 0
        cpu_device = torch.device('cpu')
        gpu_device = torch.device("cuda")
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=cpu_device)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.tasks = checkpoint['tasks']
            self.gpu = checkpoint['gpu']
            self.model_dir = checkpoint['model_dir']
            epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint['net'])
            if torch.cuda.is_available() and self.gpu:
                self.model.to(gpu_device)
                self.device = gpu_device
            else:
                self.device = cpu_device
            self.is_trained = True
            if not self.loss_function and 'loss' in checkpoint.keys():
                self.loss_function = checkpoint["loss"]
            self._transfer_loss_and_metrics_modules_to_right_device()
        return epoch + 1

class GANScheduler(_PyTorchLRSchedulerWrapper):
    def __init__(self, optim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim = optim

    def on_train_begin(self, logs):
        if "gen" in self.optim.lower():
            optimizer = self.model.G_optimizer
        else:
            optimizer = self.model.D_optimizer

        self.scheduler = self.torch_lr_scheduler(optimizer, *self.args, **self.kwargs)

        if self.loaded_state is not None:
            self.scheduler.load_state_dict(self.loaded_state)
            self.loaded_state = None

class GANTrainer(Trainer):
    def __init__(self, *args, n_critic=1, optimizer='adam',
                 **kwargs):  # net, loss_fn, metrics={}, optimizer='adam', gpu=False, tasks=[], model_dir="", **kwargs):
        super(GANTrainer, self).__init__(*args, optimizer=optimizer, **kwargs)
        self.n_critic = n_critic
        if isinstance(optimizer, str):
            optimizer = all_optimizers_dict[optimizer.lower()]
        if not issubclass(optimizer, optim.Optimizer):
            raise ValueError("Optimizer could not be processed")
        Dopt_params = params_getter(
            kwargs, ("D_optim__", "D_op__", "D_optimizer__"))
        Gopt_params = params_getter(
            kwargs, ("G_optim__", "G_op__", "G_optimizer__"))
        self.D_optimizer = optimizer(self.model.D_params(), **Dopt_params)
        self.G_optimizer = optimizer(self.model.G_params(), **Gopt_params)
        self.optimizer = self.G_optimizer
        self.metrics_names += ["gloss"]
        self.metrics += [lambda x: x]

    def save(self, path, epoch=1, save_loss=False):
        state = {
            'epoch': epoch,
            'net': self.network_state_dict(),
            'G_optimizer': self.G_optimizer.state_dict(),
            'D_optimizer': self.D_optimizer.state_dict(),
            'model_dir': self.model_dir,
            'gpu': self.gpu,
        }
        if save_loss:
            state.update(loss=self.loss_function)
        torch.save(state, path)

    def resume_fit(self, model_path, generator=False, **kwargs):
        epoch = self.restore(model_path)
        kwargs["initial_epoch"] = epoch
        if generator:
            return self.fit(**kwargs)
        else:
            return self.fit_generator(**kwargs)

    def restore(self, path):
        epoch = 0
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
            self.gpu = checkpoint['gpu']
            self.model_dir = checkpoint['model_dir']
            epoch = checkpoint["epoch"]
            self.is_trained = True
            if not self.loss_function and 'loss' is checkpoint.keys():
                self.loss_function = checkpoint["loss"]
        return epoch + 1

    def _transfer_optimizer_state_to_right_device(self):
        for optimizer in [self.G_optimizer, self.D_optimizer]:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p in optimizer.state:
                        for _, v in optimizer.state[p].items():
                            if torch.is_tensor(v) and p.device != v.device:
                                v.data = v.data.to(p.device)

    def _fit_batch(self, x, *y, callback=Callback(), step=None, return_pred=False):


        # =================================================================================== #
        #   Train the discriminator                              #
        # =================================================================================== #
        *x, real_mols = x
        batch_size = self._get_batch_size(*x, *y)
        z = self.model.sample(batch_size).to(self.device)
        logits_real = self.model.forward_discriminator(x, mols=real_mols)
        fake_data = self.model.postprocess(self.model.forward_generator(z))
        logits_fake = self.model.forward_discriminator(fake_data)

        d_loss = self.model.adversarial_loss(logits_real, logits_fake, x, fake_data)
        self.D_optimizer.zero_grad()
        d_loss.backward()
        self.D_optimizer.step()

        # =================================================================================== #
        #   Train the Generator                              #
        # =================================================================================== #

        # Every n_critic times update generator
        metrics = [0] * len(self.metrics_names)

        self.G_optimizer.zero_grad()
        if step % self.n_critic == 0:
            gen_data = self.model.forward_generator(z)
            # Postprocess with Gumbel softmax
            fake_data = self.model.postprocess(gen_data, hard=True)
            logits_fake = self.model.forward_discriminator(fake_data)
            fake_mols = data2mol(fake_data)
            g_loss = self.model.generator_loss(logits_fake, x, real_mols, fake_data, fake_mols)
            g_loss.backward()
            self.G_optimizer.step()
            # Misc for saving the data
            self.model.log(fake_mols, self.writer, step)
            for name, met in enumerate(self.metrics[:-1]):
                m = met(fake_mols)
                if isinstance(m, (np.ndarray, list)):
                    m = np.mean(m)
                metrics[name] = (float(m))
            metrics[-1] = float(g_loss)
        return float(d_loss), np.array(metrics), None

    def _compute_loss_and_metrics(self, x, y, *, return_loss_tensor=False, return_pred=False):
        *x, real_mols = x
        x, *y = self._process_input(x, *y)
        z = self.model.sample(self._get_batch_size(*x, *y)).to(self.device)
        # Z-to-target
        gen_data = self.model.forward_generator(z)
        fake_data = self.model.postprocess(gen_data, hard=True)
        logits_fake, _ = self.model.forward_discriminator(fake_data)
        loss_fake = -torch.mean(logits_fake)
        # Fake Reward
        fake_mols = data2mol(fake_data)
        metrics = np.array([float(metric(fake_mols)) for metric in self.metrics[:-1]] + [float(loss_fake)])

        return 0, metrics, fake_mols

    def sample_molecules(self, n_mols, batch_size=None):
        mols = []
        if not batch_size:
            batch_size = n_mols
        n_sample = n_mols // batch_size
        self.model.eval()
        with torch.no_grad():
            for k in range(n_sample):
                z = self.model.sample(batch_size).to(self.device)
                gen_data = self.model.forward_generator(z)
                # Postprocess with Gumbel softmax
                fake_data = self.model.postprocess(gen_data, hard=True)
                fake_mols = data2mol(fake_data)
                smiles = convert_mol_to_smiles(fake_mols)
                mols.extend(smiles)
        return mols

class AAETrainer(GANTrainer):
    def __init__(self, *args, n_critic=1, optimizer='adam', pretrain=False, disc_coeff=1e-3, **kwargs):
        super(AAETrainer, self).__init__(*args, optimizer=optimizer, **kwargs)
        self.disc_coeff = disc_coeff
        self._pretrain = pretrain
        self.metrics_names[-1] = "dloss"

    def set_training_mode(self, pretrain=False):
        self._pretrain = pretrain

    def _fit_batch(self, x, *y, callback=Callback(), step=None, return_pred=False):
        # =================================================================================== #
        #   Train the autoencoder                              #
        # =================================================================================== #
        *x, real_mols = x
        b_size = self._get_batch_size(*x, *y)
        z = self.model.forward_encoder(x, mols=real_mols)
        # Compute loss with real graph.

        x_hat = self.model.forward_decoder(z)
        real_D_outputs = self.model.forward_discriminator(z)
        rec_mols = data2mol(self.model.postprocess(x_hat, hard=True))
        ae_loss = (1 - self.disc_coeff) * self.model.autoencoder_loss(z, x, real_mols, x_hat,
                                                                      rec_mols) + self.disc_coeff * self.model.oneside_discriminator_loss(
            real_D_outputs, truth=True)

        self.G_optimizer.zero_grad()
        ae_loss.backward(retain_graph=True)
        self.G_optimizer.step()

        # =================================================================================== #
        #   Train the discriminator                              #
        # =================================================================================== #
        d_loss = 0
        if not self._pretrain:
            self.model.encoder.eval()
            self.D_optimizer.zero_grad()
            z_pre = self.model.sample(b_size).to(self.device)
            real_D_outputs = self.model.forward_discriminator(z)
            fake_D_outputs = self.model.forward_discriminator(z_pre)
            d_loss = self.model.discriminator_loss(real_D_outputs, fake_D_outputs)
            d_loss.backward()
            self.D_optimizer.step()

        self.model.log(real_mols, rec_mols, self.writer, step)
        # Every n_critic times update generator
        metrics = [0] * len(self.metrics_names)
        # Misc for saving the data
        for name, met in enumerate(self.metrics[:-1]):
            m = met(rec_mols)
            if isinstance(m, (np.ndarray, list)):
                m = np.mean(m)
            metrics[name] = (float(m))
        metrics[-1] = float(d_loss)
        return float(ae_loss), np.array(metrics), None

    def _compute_loss_and_metrics(self, x, y, *, return_loss_tensor=False, return_pred=False):
        *x, real_mols = x
        x, *y = self._process_input(x, *y)
        b_size = self._get_batch_size(x, *y)
        z = self.model.forward_encoder(x, mols=real_mols)
        # Compute loss with real graph.
        x_hat = self.model.forward_decoder(z)
        # Z-to-target
        rec_data = self.model.postprocess(x_hat, hard=True)
        # Fake Reward
        rec_mols = data2mol(rec_data)
        loss = self.model.autoencoder_loss(z, x, real_mols, x_hat, rec_mols)
        metrics = np.array([float(metric(rec_mols)) for metric in self.metrics[:-1]] + [float(loss)])
        return float(loss), metrics, rec_mols

    def sample_molecules(self, n_mols, batch_size=None):
        mols = []
        if not batch_size:
            batch_size = n_mols
        n_sample = n_mols // batch_size
        self.model.eval()
        with torch.no_grad():
            for k in range(n_sample):
                z = self.model.sample(batch_size).to(self.device)
                gen_data = self.model.forward_decoder(z)
                # Postprocess with Gumbel softmax
                fake_data = self.model.postprocess(gen_data, hard=True)
                fake_mols = data2mol(fake_data)
                smiles = convert_mol_to_smiles(fake_mols)
                mols.extend(smiles)
        return mols

    def predict_generator(self, dataset, *, steps=None):
        if steps is None and hasattr(generator, '__len__'):
            steps = len(generator)
        pred_y = []
        self.model.eval()
        with torch.no_grad():
            for _, (x, *y) in _get_step_iterator(steps, generator):
                x, *y = self._process_input(x, *y)
                pred_y.append(torch_to_numpy(self.model(x)))
        return np.concatenate(pred_y)

class SupervisedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SupervisedTrainer, self).__init__(*args, **kwargs)

    def restore(self, path):
        r"""
        Restore the training state (model, optimizer) saved in path, inside the current
        instance of the class, then return the next epoch number. If a loss function is
        not currently defined, it is loaded from the pickle object.

        Arguments
        ----------
            path: str
                path to the file containing the serialized model.

        Returns
        -------
            The last epoch saved + 1
        """
        epoch = 0
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.tasks = checkpoint['tasks']
            self.gpu = checkpoint['gpu']
            self.model_dir = checkpoint['model_dir']
            epoch = checkpoint["epoch"]
            self.is_trained = True
            if not self.loss_function and 'loss' is checkpoint.keys():
                self.loss_function = checkpoint["loss"]
        return epoch + 1


    def _compute_metrics(self, pred_y, y):
        if isinstance(pred_y, tuple):
            pred_y = pred_y[0]
        
        if pred_y.shape[-1] == 1:
            pred_y = torch.sigmoid(pred_y)
        else:
            pred_y = torch.softmax(pred_y, dim=1)
        return np.array([float(metric(pred_y.detach(), *y)) for metric in self.metrics])
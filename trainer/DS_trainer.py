import logging
from typing import Any, Dict, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from utils.batch_calculate_index import calculate_MI, calculate_conditional_MI
import math

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader

log = logging.getLogger('torcheeg')


def classification_metrics(metric_list: List[str], num_classes: int):
    allowed_metrics = [
        'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc',
        'kappa', 'specificity'
    ]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', 'kappa', 'specificity'."
            )

    class Specificity(torchmetrics.Metric): 
        def __init__(self, num_classes: int, dist_sync_on_step: bool = False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.num_classes = num_classes
            self.add_state("true_negative", default=torch.zeros(1))
            self.add_state("false_positive", default=torch.zeros(1))

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            confusion = torchmetrics.functional.confusion_matrix(preds.argmax(-1), target, task='binary')
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            self.true_negative += TN
            self.false_positive += FP

        def compute(self):
            specificity = self.true_negative / (self.true_negative + self.false_positive)
            return specificity.mean()  

    metric_dict = {
        'accuracy':
            torchmetrics.Accuracy(task='multiclass',
                                  num_classes=num_classes,
                                  top_k=1),
        'precision':
            torchmetrics.Precision(task='multiclass',
                                   average='macro',
                                   num_classes=num_classes),
        'recall':
            torchmetrics.Recall(task='multiclass',
                                average='macro',
                                num_classes=num_classes),
        'f1score':
            torchmetrics.F1Score(task='multiclass',
                                 average='macro',
                                 num_classes=num_classes),
        'matthews':
            torchmetrics.MatthewsCorrCoef(task='multiclass',
                                          num_classes=num_classes),
        'auroc':
            torchmetrics.AUROC(task='multiclass', num_classes=num_classes),
        'kappa':
            torchmetrics.CohenKappa(task='multiclass', num_classes=num_classes),
        'specificity':  
            Specificity(num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG classification.

        .. code-block:: python

            trainer = ClassifierTrainer(model)
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
            num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', and 'kappa'. (default: :obj:`["accuracy"]`)

        .. automethod:: fit
        .. automethod:: test
    '''

    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super().__init__()
        self.model = model

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.CrossEntropyLoss()

        self.init_metrics(metrics, num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(trainable_parameters,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        return y_hat


class NormLoss(nn.Module):
    def __init__(self, norm="l1"):
        super(NormLoss, self).__init__()
        assert norm in ["l1", "l2", "none"]
        self.norm = norm

    def forward(self, x):
        if self.norm == "l1":
            return torch.sum(torch.abs(x))
        elif self.norm == "l2":
            return torch.sum(torch.pow(x, 2))
        else:
            return None


class SingleTrainer(ClassifierTrainer):
    def __init__(self, classifier, ixy, ci,
                 label_smoothing: float = 0.1,
                 Nalpha: int = 56, args=None, lambda_disc=0.01, lambda_norm=0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = self.model
        self.classifier = classifier
        self.save_hyperparameters(ignore=["model", "classifier"])
        self.ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.Nalpha = Nalpha
        self.ixy = ixy
        self.ci = ci
        self.args = args
        self.norm = NormLoss(norm=args.norm)

        self.lambda_disc = lambda_disc
        self.lambda_norm = lambda_norm

    def main(self, batch):
        x, y = batch
        finalx, alpha, beta, mask, disc_fake, disc_ture = self.encoder(x, self.args.reverse)
        return finalx, alpha, beta, mask, disc_fake, disc_ture

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch

        current_epoch = self.trainer.current_epoch
        train_batches = len(self.trainer.train_dataloader)
        p = (current_epoch * train_batches + batch_idx) / (self.args.max_epoch * train_batches)
        m = 2. / (1. + math.exp(-1 * p)) - 1

        finalx, alpha, beta, mask, disc_fake, disc_true = self.main(batch)  # (16, 64)

        normloss = self.norm(mask)
        disc_loss = (self.ce_fn(disc_true, torch.ones(disc_true.size(0), dtype=torch.long, device=disc_true.device)) +
                     self.ce_fn(disc_fake, torch.zeros(disc_fake.size(0), dtype=torch.long, device=disc_fake.device)))

        Ixy = calculate_MI(alpha, beta)
        CI = calculate_conditional_MI(alpha, y.float(), beta)

        y_hat = self.classifier(finalx)
        celoss = self.ce_fn(y_hat, y)

        if normloss is None:
            loss = celoss + m * (- self.ci * CI) + self.lambda_disc * disc_loss
        else:
            loss = celoss + m * (- self.ci * CI) + self.lambda_norm * normloss + self.lambda_disc * disc_loss  # full
          
        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=True,
                 logger=False,  # modi
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            if self.metrics[i] == "accuracy":
                self.log(f"train_{self.metrics[i]}",
                         metric_value(y_hat, y),
                         prog_bar=True,
                         on_epoch=True,
                         logger=False,  # modi
                         on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            if self.metrics[i] == "accuracy":
                self.log(f"train_{self.metrics[i]}",
                         metric_value.compute(),
                         prog_bar=False,
                         on_epoch=True,
                         on_step=False,
                         logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        finalx, alpha, beta, mask, disc_fake, disc_true = self.main(batch)  # (16, 64)

        # alpha_beta = torch.cat((alpha, beta), dim=1)
        y_hat = self.classifier(finalx)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            if self.metrics[i] == "accuracy":
                self.log(f"val_{self.metrics[i]}",
                         metric_value.compute(),
                         prog_bar=True,
                         on_epoch=True,
                         on_step=False,
                         logger=True)

        # print the metrics
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        finalx, alpha, beta, mask, disc_fake, disc_true = self.main(batch)  # (16, 64)

        # y_hat = self.classifier(alpha)
        # alpha_beta = torch.cat((alpha, beta), dim=1)
        y_hat = self.classifier(finalx)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()



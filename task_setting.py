import pytorch_lightning as pl
import torch
import torchmetrics


class Task(pl.LightningModule):
    def __init__(
        self,
        name="energy_U0",
        model=None,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-5},
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.metrics = torch.nn.ModuleDict(metrics)
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args

        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return self.model(batch)

    def _compute_loss_and_metrics(self, batch, batch_idx, stage):
        targets = {self.name: batch[self.name]}
        pred = self(batch) 

        loss = self.loss_weight * self.loss_fn(pred[self.name], targets[self.name])

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=(stage != "train"), prog_bar=True)

        if stage == "val":
            self.model.forward(batch, val_loss=loss.item())

        for name, metric in self.metrics.items():
            metric_value = metric(pred[self.name], targets[self.name])
            self.log(f"{stage}_{name}", metric_value, on_step=(stage == "train"), on_epoch=(stage != "train"), prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, "test")

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), **self.optimizer_args)

    def on_train_epoch_end(self):
        self.model.all_molecular_features.clear()

    def save_model(self, path: str):
        if self.global_rank == 0:
            torch.save(self.model, path)

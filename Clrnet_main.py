import os
import schnetpack as spk
import schnetpack.transform as trn
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from energy_regression import Clrnet
from feature_extraction import Cfconv
from task_setting import Task

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
save_path = './model_save'
if not os.path.exists('model_save'):
    os.makedirs(save_path)

custom_data = spk.data.AtomsDataModule(
    datapath='./your_database_name.db',
    batch_size=20,
    distance_unit='Ang',
    property_units={"energy_U0":'eV'},
    num_train=0.8,
    num_val=0.1,
    num_test=0.1,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets("energy_U0", remove_mean=True),
        trn.CastTo32()
    ],
    num_workers=8,
    pin_memory=True,  # set to false, when not using a GPU
)

if __name__ == '__main__':
    custom_data.prepare_data()
    custom_data.setup()


    cutoff = 5.
    n_atom_basis = 32
    n_rbf = 20

    fea_extr = Cfconv(
        n_atom_basis=n_atom_basis, n_interactions=3,
        n_rbf=n_rbf, cutoff=cutoff
    )

    ene_reg = Clrnet(n_atom_basis=n_atom_basis,feature_model=fea_extr,output_key="energy_U0")

    task = Task(
        model=ene_reg,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 5e-5}
    )

    logger = pl.loggers.CSVLogger(save_dir='logs', name='log')

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename="best_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            verbose=True,
        ),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=save_path,
        max_epochs=100,
    )

    trainer.fit(task, datamodule=custom_data)

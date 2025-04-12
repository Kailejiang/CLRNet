import os
import yaml
import argparse
import schnetpack as spk
import schnetpack.transform as trn
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from energy_regression import Clrnet
from feature_extraction import Cfconv
from task_setting import Task


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Parse command-line arguments for overriding configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with SchNetPack")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration file")
    parser.add_argument('--datapath', type=str, help="Path to the database")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for the optimizer")
    parser.add_argument('--max_epochs', type=int, help="Maximum number of training epochs")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    # Override configuration with command-line arguments (if provided)
    if args.datapath:
        config['data']['datapath'] = args.datapath
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs

    # Set up paths
    save_path = config['training']['save_path']
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up data module
    custom_data = spk.data.AtomsDataModule(
        datapath=config['data']['datapath'],
        batch_size=config['data']['batch_size'],
        distance_unit=config['data']['distance_unit'],
        property_units=config['data']['property_units'],
        num_train=config['data']['num_train'],
        num_val=config['data']['num_val'],
        num_test=config['data']['num_test'],
        transforms=[
            trn.ASENeighborList(cutoff=config['data']['cutoff']),
            trn.RemoveOffsets("energy_U0", remove_mean=True),
            trn.CastTo32()
        ],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )

    custom_data.prepare_data()
    custom_data.setup()

    # Set up model components
    fea_extr = Cfconv(
        n_atom_feature=config['model']['n_atom_feature'],
        n_interactions=config['model']['n_interactions'],
        n_distance_feature=config['model']['n_distance_feature'],
        cutoff=config['data']['cutoff']
    )

    ene_reg = Clrnet(
        n_atom_basis=config['model']['n_atom_feature'],
        feature_model=fea_extr,
        output_key="energy_U0"
    )

    task = Task(
        model=ene_reg,
        optimizer_cls=getattr(torch.optim, config['training']['optimizer']),
        optimizer_args={"lr": config['training']['learning_rate']}
    )

    # Set up logger and callbacks
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
            patience=config['training']['patience'],
            mode="min",
            verbose=True,
        ),
    ]

    # Set up trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=save_path,
        max_epochs=config['training']['max_epochs'],
    )

    trainer.fit(task, datamodule=custom_data)

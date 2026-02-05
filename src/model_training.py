import pytorch_lightning as pl


def get_simple_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.simple_model import SimpleClassifier
    params = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 256,
        "output_dim": 2,
        "n_layers": 1,
        "vocab_size": vocab_size,
        "max_length": max_length,
        "optimizer": {"name": "Adam",
                      "lr": 1e-4},
        "loss_type": "cross_entropy",
    }
    params.update(kwargs)
    return SimpleClassifier(**params), "simple"


def get_rnn_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.rnn_simple_model import RNNClassifier
    params = {
        "vocab_size": vocab_size,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 512,
        "output_dim": 2,
        "n_layers": 1,
        "max_length": max_length,
        "optimizer": {"name": "Adam",
                      "lr": 2e-3},
        "loss_type": "cross_entropy",
    }
    params.update(kwargs)
    return RNNClassifier(**params), "rnn"


def get_lstm_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.lstm_model import LSTMClassifier
    params = {
        "vocab_size": vocab_size,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 18,
        "hidden_dim2": 12,
        "output_dim": 2,
        "n_layers": 1,
        "bidirectional": False,
        "dropout": 0.75,
        "max_length": max_length,
        "optimizer": {"name": "Adam",
                      "lr": 6e-4},
        "loss_type": "cross_entropy",
        "dataset_version": "v2"
    }
    params.update(kwargs)
    return LSTMClassifier(**params), "lstm"


def get_bert_model(batch_size, vocab_size, max_length, max_epochs, **kwargs):
    from models.bert_model import BERTClassifier
    params = {
        "vocab_size": vocab_size,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedding_dim": 350,
        "hidden_dim": 512,
        "output_dim": 2,
        "n_layers": 1,
        # "max_length": max_length,
        "optimizer": {"name": "AdamW",
                      "lr": 2e-5},
        "loss_type": "cross_entropy",
        "model_name": kwargs.get("model_name", "bert-base-uncased"),
    }
    params.update(kwargs)
    return BERTClassifier(**params), "bert"


if __name__ == "__main__":
    import argparse
    from pytorch_lightning.loggers import TensorBoardLogger
    from dataset.loader import get_loaders

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn",
                        choices=["simple", "rnn", "lstm", "bert"])
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--tokenizer_name", type=str,
                        default="distilbert-base-uncased")
    parser.add_argument("--eval_only", type=int, default=0)
    args = parser.parse_args()

    use_clearml = True
    continue_last_model = False

    project_name = "MDPI"
    if use_clearml:
        from clearml import Task
        from clearml.backend_api.session.defs import MissingConfigError

    # Seed
    seed = 0
    pl.seed_everything(seed)

    # Params
    max_length = 110
    batch_size = 64
    max_epochs = 100
    tokenizer_name = args.tokenizer_name

    # Load data
    train_loader, val_loader, test_loader, vocab_size = get_loaders(
        "dataset/sentence/train.csv",
        "dataset/sentence/val.csv",
        "dataset/sentence/test.csv",
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        max_length=max_length
    )

    # --------- MODEL SELECTION (THIS IS WHAT YOU ASKED) ----------
    if args.model == "simple":
        model, tag = get_simple_model(batch_size, vocab_size, max_length, max_epochs, tokenizer_name=tokenizer_name)
    elif args.model == "rnn":
        model, tag = get_rnn_model(batch_size, vocab_size, max_length, max_epochs, tokenizer_name=tokenizer_name)
    elif args.model == "lstm":
        model, tag = get_lstm_model(batch_size, vocab_size, max_length, max_epochs, tokenizer_name=tokenizer_name)
    else:  # bert
        model, tag = get_bert_model(batch_size, vocab_size, max_length, max_epochs, model_name=tokenizer_name)
    # -------------------------------------------------------------

    # Trainer (force device choice)
    accelerator = "gpu" if args.device == "cuda" else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=TensorBoardLogger("./tb_logs", name=model.name),
        accelerator=accelerator,
        devices=1
    )

    # ClearML (optional)
    task = None
    if use_clearml:
        try:
            task = Task.init(
                project_name=project_name,
                task_name=model.name,
                tags=[tag],
                continue_last_task=continue_last_model
            )
            task.set_model_config(config_text=str(model))
        except MissingConfigError:
            print("ClearML not configured, continuing without ClearML.")
            task = None

    # Train + test
    if not args.eval_only:
        trainer.fit(model, train_loader, val_dataloaders=val_loader)

    trainer.test(model, test_loader)

    if task is not None:
        task.close()
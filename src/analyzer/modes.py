from chess_utils import fen_to_tensor
from my_torch import Network

from .data_loader import load_chessfile_predict, load_chessfile_train
from .labels import vector_to_label


def predict_mode(
    network: Network, chessfile: str, encoding: str = "simple"
) -> None:
    fens = load_chessfile_predict(chessfile)

    for fen in fens:
        try:
            x = fen_to_tensor(fen, encoding=encoding)
            output = network.predict(x)
            label = vector_to_label(output)
            print(label)
        except Exception as e:
            print(f"Error: {e}")


def train_mode(
    network: Network,
    chessfile: str,
    savefile: str,
    epochs: int = 1000,
    encoding: str = "simple",
) -> None:
    print(f"Loading training data from {chessfile}...")
    dataset = load_chessfile_train(chessfile, encoding=encoding)

    if not dataset:
        print("Error: No training data loaded")
        return

    print(f"Loaded {len(dataset)} training samples")
    print(
        f"Input size: {len(dataset[0][0])}, Output size: {len(dataset[0][1])}"
    )

    val_split = int(len(dataset) * 0.2)
    val_data = dataset[:val_split]
    train_data = dataset[val_split:]

    print(
        f"Training on {len(train_data)} samples, validating on {len(val_data)}"
    )
    print("Starting training...")

    history = network.train(
        train_data,
        epochs=epochs,
        target_accuracy=0.95,
        batch_size=64,
        validation_data=val_data,
        verbose=True,
    )

    train_acc = network.evaluate(train_data)
    val_acc = network.evaluate(val_data)
    print(f"\nFinal: Train={train_acc*100:.2f}% Val={val_acc*100:.2f}%")
    print(f"Saving network to {savefile}...")
    network.save(savefile)
    print("Done!")

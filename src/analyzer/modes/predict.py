import functools
from concurrent.futures import ProcessPoolExecutor

from chess_utils import fen_to_tensor
from my_torch import Network

from ..data_loader import load_chessfile_predict
from ..labels import vector_to_label
from ..profile import profile_it


def process_fen(network: Network, encoding: str, fen: str):
    output = network.predict(fen_to_tensor(fen, encoding=encoding))
    return vector_to_label(output)


def predict_mode(
    network: Network, chessfile: str, encoding: str = "simple"
) -> None:
    fens = load_chessfile_predict(chessfile)

    dispatch = functools.partial(process_fen, network, encoding)

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(dispatch, fens, chunksize=1024)

        for result in results:
            print(result)


@profile_it
def predict_mode_profiled(
    network: Network, chessfile: str, encoding: str = "simple"
) -> None:
    fens = load_chessfile_predict(chessfile)

    dispatch = functools.partial(process_fen, network, encoding)

    for c, fen in enumerate(fens):
        print(c, dispatch(fen))

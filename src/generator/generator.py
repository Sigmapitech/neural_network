import pathlib

from .config_loader import load_config
from .network_builder import create_network_from_config


def generate_networks(config_file: str, count: int) -> None:
    print(f"Loading config from {config_file}...")
    config = load_config(config_file)

    base_name = config.get("name", pathlib.Path(config_file).stem)

    for i in range(1, count + 1):
        config_copy = config.copy()
        if "seed" in config_copy:
            config_copy["seed"] = config_copy["seed"] + i
        else:
            config_copy["seed"] = i

        network = create_network_from_config(config_copy)

        output_file = f"{base_name}_{i}.nn"
        network.save(output_file)
        print(f"Created {output_file}")

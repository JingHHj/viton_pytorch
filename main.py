from train import Trainer 
import argparse
import yaml


def load_config():
    # Load YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    config = load_config()

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
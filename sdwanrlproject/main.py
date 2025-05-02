from core.controller import SDWANController

if __name__ == "__main__":
    controller = SDWANController(
        env_config="sdwanrlproject/configs/topology.yaml",
        train_config="sdwanrlproject/configs/training.yaml"
    )
    controller.train()
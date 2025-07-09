# TODO [Phase 7]: Implement training loop
class PretrainTrainer:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config

    def train(self):
        for epoch in range(self.config["num_epochs"]):
            # TODO: Train model for one epoch
            print(f"Epoch {epoch}: TODO")
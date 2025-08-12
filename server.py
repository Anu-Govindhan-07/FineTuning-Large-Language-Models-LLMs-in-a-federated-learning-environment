import flwr as fl
from flwr.server.strategy import FedAvg
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils.model_utils import get_model, save_model
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG)


class SaveModelStrategy(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
            model.load_state_dict(dict(zip(model.state_dict().keys(), [torch.tensor(w) for w in aggregated_weights if w is not None])))
            save_model(model, f"model_round_{rnd}.pth")
            logging.info(f"Model for round {rnd} saved.")
        return aggregated_weights


def start_server():
    logging.info("Loading pre-trained model...")
    model = get_model()
    model_path = "model_initial.pth"
    save_model(model, model_path)
    logging.info(f"Model loaded from {model_path}")

    strategy = SaveModelStrategy(
        min_available_clients=2,
        min_fit_clients=2,
        min_eval_clients=2,
        on_fit_config_fn=lambda rnd: {"round": rnd},
    )

    logging.info("Starting Flower server...")
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config={"num_rounds": 3},
            strategy=strategy
        )
    except Exception as e:
        logging.error(f"Error starting server: {e}", exc_info=True)


if __name__ == "__main__":
    start_server()

import flwr as fl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
from utils.model_utils import get_model
from utils.data_utils import preprocess_data, load_and_split_data
from rouge_score import rouge_scorer
import os
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)


class SummarizationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SummarizationClient(fl.client.NumPyClient):
    def __init__(self, model, dataloader, tokenizer, client_id):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.client_id = client_id
        self.optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.scores_file = Path(f"client_{self.client_id}_scores.json")

    def get_parameters(self):
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        logging.info(f"[CLIENT {self.client_id}] get_parameters called")
        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        logging.info(f"[CLIENT {self.client_id}] set_parameters called")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0
        logging.info(f"[CLIENT {self.client_id}] Training started")
        try:
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                logging.debug(f"[CLIENT {self.client_id}] Processing batch: {batch}")
                inputs = preprocess_data(batch, self.tokenizer)
                logging.debug(f"[CLIENT {self.client_id}] Preprocessed inputs: {inputs}")
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logging.info(f"[CLIENT {self.client_id}] Training completed, total loss: {total_loss}")
        except Exception as e:
            logging.error(f"[CLIENT {self.client_id}] Error during training: {e}", exc_info=True)

        model_path = f"model_client_{self.client_id}.pth"
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"[CLIENT {self.client_id}] Model saved as {model_path}")

        return self.get_parameters(), len(self.dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0
        rouge1, rouge2, rougel = 0, 0, 0
        num_batches = 0
        try:
            with torch.no_grad():
                logging.info(f"[CLIENT {self.client_id}] Evaluation started")
                for batch in self.dataloader:
                    inputs = preprocess_data(batch, self.tokenizer)
                    outputs = self.model.generate(input_ids=inputs['input_ids'],
                                                  attention_mask=inputs['attention_mask'], max_length=150)
                    preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    refs = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)
                    for pred, ref in zip(preds, refs):
                        scores = self.scorer.score(ref, pred)
                        rouge1 += scores['rouge1'].fmeasure
                        rouge2 += scores['rouge2'].fmeasure
                        rougel += scores['rougeL'].fmeasure
                    num_batches += 1
            avg_rouge1 = rouge1 / num_batches
            avg_rouge2 = rouge2 / num_batches
            avg_rougel = rougel / num_batches
            logging.info(
                f"[CLIENT {self.client_id}] Evaluation completed, average loss: {total_loss / num_batches}, ROUGE-1: {avg_rouge1}, ROUGE-2: {avg_rouge2}, ROUGE-L: {avg_rougel}")

            round_scores = {"round": config["round"], "rouge1": avg_rouge1, "rouge2": avg_rouge2, "rougel": avg_rougel}
            if self.scores_file.exists():
                with open(self.scores_file, "r") as f:
                    all_scores = json.load(f)
            else:
                all_scores = []

            all_scores.append(round_scores)

            with open(self.scores_file, "w") as f:
                json.dump(all_scores, f)

            return total_loss / num_batches, len(self.dataloader.dataset), {"rouge1": avg_rouge1, "rouge2": avg_rouge2,
                                                                            "rougel": avg_rougel}
        except Exception as e:
            logging.error(f"[CLIENT {self.client_id}] Error during evaluation: {e}", exc_info=True)
            return total_loss / num_batches, len(self.dataloader.dataset), {"rouge1": 0, "rouge2": 0, "rougel": 0}


def main(client_id):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = get_model()

    model_path = f"model_client_{client_id}.pth"
    if os.path.exists(model_path):
        logging.info(f"[CLIENT {client_id}] Loading existing model for client {client_id}")
        model.load_state_dict(torch.load(model_path))
    else:
        logging.info(f"[CLIENT {client_id}] Training new model for client {client_id}")

    dataset = load_and_split_data(client_id)
    dataloader = DataLoader(SummarizationDataset(dataset), batch_size=16, shuffle=True)

    client = SummarizationClient(model, dataloader, tokenizer, client_id)

    logging.info(f"[CLIENT {client_id}] Client {client_id} connecting to server at localhost:8080")
    fl.client.start_numpy_client("localhost:8080", client=client)


if __name__ == "__main__":
    import sys

    client_id = int(sys.argv[1])
    main(client_id)

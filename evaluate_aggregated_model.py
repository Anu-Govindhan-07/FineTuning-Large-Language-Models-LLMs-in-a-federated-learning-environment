import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
from rouge_score import rouge_scorer
from utils.data_utils import load_local_data, preprocess_data
from utils.model_utils import load_model
import logging
import json
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


class SummarizationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_model(model, dataloader, tokenizer):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougel = 0, 0, 0
    num_batches = 0
    with torch.no_grad():
        logging.info("Evaluation started")
        for batch in dataloader:
            inputs = preprocess_data(batch, tokenizer)
            outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)
            for pred, ref in zip(preds, refs):
                scores = scorer.score(ref, pred)
                rouge1 += scores['rouge1'].fmeasure
                rouge2 += scores['rouge2'].fmeasure
                rougel += scores['rougeL'].fmeasure
            num_batches += 1
    avg_rouge1 = rouge1 / num_batches
    avg_rouge2 = rouge2 / num_batches
    avg_rougel = rougel / num_batches
    logging.info(
        f"Evaluation completed, ROUGE-1: {avg_rouge1}, ROUGE-2: {avg_rouge2}, ROUGE-L: {avg_rougel}")

    return {"rouge1": avg_rouge1, "rouge2": avg_rouge2, "rougel": avg_rougel}


def main():
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    test_data_file = "dataset/medical_dataset.json"
    test_data = load_local_data(test_data_file)
    test_dataloader = DataLoader(SummarizationDataset(test_data), batch_size=16, shuffle=False)

    round_scores = []

    for round_num in range(1, 4):  # Assuming 3 rounds, adjust if needed
        model_path = f"aggregated_model_round_{round_num}.pth"
        model = load_model(model_path, model_name)
        scores = evaluate_model(model, test_dataloader, tokenizer)
        scores["round"] = round_num
        round_scores.append(scores)

    with open("aggregated_model_scores.json", "w") as f:
        json.dump(round_scores, f)

    rounds = [score["round"] for score in round_scores]
    rouge1 = [score["rouge1"] for score in round_scores]
    rouge2 = [score["rouge2"] for score in round_scores]
    rougel = [score["rougel"] for score in round_scores]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, rouge1, label="ROUGE-1", marker='o')
    plt.plot(rounds, rouge2, label="ROUGE-2", marker='o')
    plt.plot(rounds, rougel, label="ROUGE-L", marker='o')
    plt.xlabel("Round")
    plt.ylabel("ROUGE Score")
    plt.title("ROUGE Scores of Aggregated Model")
    plt.legend()
    plt.grid(True)
    plt.savefig("aggregated_model_rouge_scores.png")
    plt.show()


if __name__ == "__main__":
    main()

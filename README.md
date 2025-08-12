Federated Text Summarization using T5 Model

This project explores the application of federated learning for fine-tuning a large language model (T5-small) for text summarization in the healthcare sector.
The objective is to evaluate whether federated learning can enhance model performance while preserving data privacy across multiple decentralized clients.


--Table of Contents--
>Introduction
>Project Structure
>Setup and Installation
>Dataset
>Running the Project
>Technologies Used



>Introduction 

This project aims to fine-tune a T5 model for text summarization using federated learning, ensuring data privacy and security in the healthcare sector.
The federated learning framework Flower is used to manage the communication and aggregation of model updates across multiple decentralized clients


Project Structure
federated_text_summarization_T5/
├── client/
│   ├── client.py
├── server/
│   ├── server.py
├── utils/
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── __init__.py
├── dataset/
│   ├── medical_meadow_cord19.json
├── web/
│   ├── app.py
│   ├── static/
│   │   ├── script.js
│   │   ├── style.css
│   ├── templates/
│   │   ├── index.html
│──plot_decentralized_rouge_scores.py
├──evaluate_aggregated_model.py
├── README.md
└── requirements.txt

> Setup and Installation<

pip install -r requirements.txt

>Dataset
> 
Download the dataset: https://huggingface.co/datasets/medalpaca/medical_meadow_cord19?row=13


>Running the Project

Run each command separate terminals


Create a virtual environment:
1.python -m venv env
source env/bin/activate  # On Windows: `env\Scripts\activate`
   

Starting the server :
1.python -m server.server


Starting the Clients
2.python -m client.client 0
3.python -m client.client 1
4.python -m client.client 2

ROUGE Score Evaluation
5.python -m evaluate_aggregated_model

Starting the web Application
6.python -m web.app
7.python -m web.app
8.python -m web.app



Running the Web Application

Access the web interface at http://127.0.0.1:5000.
Access the web interface at http://127.0.0.1:5001.
Access the web interface at http://127.0.0.1:5002.


>Technologies Used

PyTorch: For model training and optimization.
Flower: Federated learning framework to manage client-server communication.
Hugging Face Transformers: For using the T5 model.
Flask: To deploy the web application.
PyCharm IDE: For development and debugging.
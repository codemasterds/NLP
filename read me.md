## Category Detection using Amazon Queries

### Overview
The project focuses on developing a Category Detection system for Amazon queries, utilizing the bert-base-uncased model, a variant of the BERT architecture pre-trained on uncased English text. The primary goal is to accurately classify user queries into specific categories to enhance the Amazon user experience by providing more relevant search results.

### Features
Category Detection: The project employs the bert-base-uncased model to predict the category of Amazon queries accurately.
PyTorch Implementation: Leveraging PyTorch, a popular deep learning library, to build and train the BERT-based model.
Kaggle Dataset: The dataset for training and evaluation is sourced from Kaggle, providing a diverse set of Amazon queries with corresponding category labels.

### Technology Stack
PyTorch: Deep learning framework used for model development and training.
BERT (bert-base-uncased): Pre-trained transformer model for natural language understanding.
Python: Programming language for implementing the model and project components.
Kaggle: Platform for data collection and exploration.

### Data Collection
Kaggle Amazon Query Dataset: The dataset, sourced from Kaggle, contains a diverse set of Amazon queries with corresponding category labels.

### Data Preprocessing: 
Cleaning and preprocessing of raw text data to prepare it for training.

### Model Architecture
BERT-based Model (bert-base-uncased): The core of the project is a deep learning model based on the BERT architecture, fine-tuned for the specific task of category detection.
Transfer Learning: Utilizing pre-trained weights of bert-base-uncased to enhance performance.

### Training
Data Splitting: The dataset is split into training and testing sets for model evaluation.
Model Training: Leveraging PyTorch to train the BERT-based model on the Kaggle-sourced dataset.

### Results
Performance Metrics: Evaluation of the model's performance using accuracy, precision, recall, and F1-score.
Kaggle Dataset Validation: Testing the model on the Kaggle dataset to ensure its applicability.

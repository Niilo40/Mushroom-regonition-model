# Mushroom Detection ML Project

This project leverages machine learning techniques to detect mushrooms based on
input data. It includes steps for data preprocessing, model training, and
evaluation, making it easy to adapt for similar classification tasks.

## Features

- Data preprocessing pipeline
- Machine learning model training
- Evaluation metrics for performance analysis
- Customizable configurations

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd ML-project-Mushroom-detection/ML-project
   ```

2. **Prepare the Environment**:\
   Install the required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preprocessing**:\
   Place your dataset in the `data/` directory. You can use the
   download_dataset.py to get the dataset.

4. **Train the Model**:\
   Before you run the model you can run the delete_photos.py if you want to make
   the dataset smaller for faster learning. Run the training script:
   ```bash
   python train.py
   ```
   Choose the model that you would like to use and add it to the train_..._.py
   name.

5. **Evaluate the Model**:\
   Use the evaluation script to analyze model performance:
   ```bash
   python evaluate.py
   ```

## File Structure

- `data/`: Directory for input datasets.
- `train.py`: Script for training the machine learning model.
- `evaluate.py`: Script for evaluating the trained model.
- `requirements.txt`: List of dependencies.

Follow these steps to adapt the project for your own classification tasks.

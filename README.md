Movie Rating Prediction Pipeline

This project demonstrates a machine learning pipeline to predict movie audience ratings. The pipeline involves data preprocessing, model training, evaluation, and saving the trained model.

## Features
1. **Data Preprocessing**:
   - Cleans the dataset by removing rows with missing or anomalous values.
   - Applies Z-score normalization to filter out outliers.
   - Extracts features for model training.

2. **Model Training**:
   - Splits the dataset into training and testing sets.
   - Scales feature values using `StandardScaler`.
   - Trains a Linear Regression model.

3. **Model Evaluation**:
   - Evaluates the model's performance using R² and distribution plots.
   - Visualizes the comparison between actual and predicted values.

4. **Model Saving**:
   - Saves the trained model using `joblib` for future use.

## Prerequisites
- Python 3.8 or higher
- Required libraries: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `joblib`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## File Structure
- `movie_pipeline.ipynb`: Jupyter Notebook with the entire pipeline.
- `dataset/Rotten_Tomatoes_Movies3.xls`: Dataset used for training and evaluation.
- `saved_model.pkl`: Saved model file (created after running the pipeline).

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Run the notebook to execute the pipeline steps:
   - Data preprocessing
   - Model training and evaluation
   - Save the model

3. Output:
   - R² Score and visualization comparing actual vs. predicted values.
   - `saved_model.pkl` will be generated in the working directory.

## Pipeline Steps
1. **Load and Preprocess Data**:
   - Function: `load_and_preprocess(file_path)`
2. **Train the Model**:
   - Function: `train_model(X, y)`
3. **Evaluate the Model**:
   - Function: `evaluate_model(model, X_test, y_test)`
4. **Save the Model**:
   - Function: `save_model(model)`

## Visualization
The evaluation step produces a plot comparing the distributions of actual and predicted ratings, helping to assess model accuracy visually.

## Contributions
Contributions to enhance the pipeline (new models, improved preprocessing, etc.) are welcome. Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

---

Let me know if you'd like any changes or additional sections in the `README.md`.

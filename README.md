# recruitment-optimization
This is the data-mining UPB course final project. This project consists in optimizing personal recuitment in building industry.

How to setup environment using conda:
1. Install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a new conda environment using the command:
   ```
   conda create --name recruitment-optimization python=3.11.9
   ```
3. Activate the environment:
   ```
    conda activate recruitment-optimization
    ```
4. Install the required packages using the command:
    ```
    pip install -r requirements_data_processing.txt
    ```

We use Streamlit to create a web application for our project. To access the web application, please visit the link https://recruitment-optimization-8dtekd553jbdjxn3q5fgns.streamlit.app/

# Deployment

This application provides an interactive deployment tool to predict **employee attrition** (abandono de cargo) in a construction sector company using a **Perceptron classifier** trained on historical data.

## Model Description
The deployed model is a **Perceptron**, a linear binary classifier from the `scikit-learn` library.  
- It learns a linear decision boundary to separate employees likely to **leave the company** ("Abandona") from those expected to **stay** ("Permanece").  
- The model was trained on a specific set of explanatory variables (features) that must be present in any dataset used for deployment.  
- Both the model and the list of expected variables are stored in a serialized file (`perceptron_model.pkl`) for consistency during inference.

## Application Name
**Predicción de abandono de cargo en empresa del sector construcción**  
This name highlights the practical use case of the application: predicting employee attrition in the construction industry.

## Quality Validations
Before generating predictions, the app performs key data validations:  
1. **File Format Check** – Only CSV files are accepted through the uploader widget.  
2. **Column Validation** – The app compares the uploaded dataset’s columns with the expected features.  
   - If one or more required variables are missing, execution is stopped.  
   - An error message clearly indicates which columns are absent (🚨).  
3. **Execution Halt** – In case of validation failure, the pipeline is interrupted with `st.stop()` to prevent invalid predictions.

## Results
Once the dataset passes validation:  
- The application extracts the required variables and applies the Perceptron model to each record.  
- Predictions are generated as binary outcomes and translated into human-readable labels:  
  - **"Abandona"** → Employee predicted to leave.  
  - **"Permanece"** → Employee predicted to stay.  
- A final **results dataframe** is displayed, containing:  
  - All input features used for prediction.  
  - An additional column **`Predicción`** with the model’s output.  

This allows decision-makers to analyze both the input variables and the prediction for each employee in a structured format.

## Workflow Diagram

```mermaid
flowchart TD
    A[Upload CSV file] --> B{Validate file}
    B -->|Invalid format| E[Error: Only CSV accepted 🚨]
    B -->|Missing columns| F[Error: Columns not found 🚨]
    B -->|Valid data| C[Extract required variables]
    C --> D[Apply Perceptron model]
    D --> G[Generate predictions: Abandona / Permanece]
    G --> H[Display results dataframe]

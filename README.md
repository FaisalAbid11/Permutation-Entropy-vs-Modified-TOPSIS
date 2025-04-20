# Permutation-Based-Entropy-vs-Modified-TOPSIS
This project compares the performance of Random Forest model using datasets obtained from using  AHP-Modified-TOPSIS combining De-PARETO principal in  and Permutation Based Entropy Method combined with TOPSIS and and interval scale.

# Overview
This project compares the performance of the **Random Forest** algorithm using six datasets, which were derived from an original dataset using two different feature selection methods. The datasets are categorized into two groups:

1. **Permutation-Based Entropy + TOPSIS with Interval Scale**  
   Three datasets — *enthusiastic*, *behavioral*, and *distressed* — were created by applying permutation-based entropy combined with TOPSIS on the original data.

2. **AHP-Modified TOPSIS + De-Pareto Principle**  
   Another three datasets were generated using a modified AHP-TOPSIS method integrated with the De-Pareto principle.

The goal of this project is to evaluate the impact of the proposed **Modified TOPSIS method** on feature selection and to assess its effectiveness in enhancing the performance of the Random Forest model. By comparing these two approaches, the study aims to identify which method provides better feature rankings and improves the predictive accuracy of employee turnover intention.


# Repository Structure
The repository is organized as follows:
```text
Permutaion-Entropy-vs-Modified-TOPSIS/
│ ├── AHP-MODIFIED-TOPSIS-EVALUATION/
│    ├── behavioral.csv
│    ├── disteressed.csv
│    ├── enthusiastic.csv
│    └── modified_ahp_TOPSIS_20_30_50.ipynb
│
│ ├── AHP-TOPSIS-EVALUATION/
│    ├── behavioral.csv
│    ├── disteressed.csv
│    ├── enthusiastic.csv
│    └── ahp_TOPSIS_20_30_50.ipynb
│
│ ├── AHP-modified-topsis-de-pareto/
│    ├── AHP.xlsx
│    └──modified_topsis.xlsx
│
│ ├── Permutation-based-entropy/
│    ├── behavioral_employees.csv
│    ├── disteressed_employees.csv
│    ├── enthusiastic_employees.csv
│    ├── new numerical.csv
│    └── permutation_based_entropy.ipynb
│
│ ├── LICENSE
└──── README.md
```
### Description:
This repository is organized into the following directories, each serving a specific part of the comparative analysis between Permutation Entropy and Modified TOPSIS:

- **AHP-MODIFIED-TOPSIS-EVALUATION:**
  Contains CSV datasets representing different employee states (behavioral, distressed, enthusiastic) and a Jupyter notebook that uses these datasets to train classifier model and sees how well a they work when datasets are created using the AHP-Modified-Topsis+De-pareto principle.

- **AHP-TOPSIS-EVALUATION:**
  Contains CSV datasets representing different employee states (behavioral, distressed, enthusiastic) and a Jupyter notebook that uses these datasets to train classifier model and sees how well a they work when datasets are created using the AHP-Topsis+De-pareto principle.
  
- **AHP-modified-topsis-de-pareto:**
  Includes supporting Excel files used for AHP scoring and modified TOPSIS rankings.
  
- **Permutation-based-entropy:**
  This folder includes datasets formatted from entropy-based analysis and a notebook that applies permutation entropy techniques to assess feature importance across employee types using cleaned another dataset (`new numerical.csv`) ready for analysis.

- **LICENSE**
  The license under which this repository is distributed.

- **README.md**
  Main documentation file providing an overview of the project, usage instructions, and descriptions of the methodology.

##  Installation & How to Run

###  Option 1: Run on Google Colab
**1. You can run the notebooks directly in your browser using Google Colab without any installation.**
   
**Links for Each Notebook :** Click the links provided for each notebook to open it in colab:
- [AHP-Modified-TOPSIS](https://colab.research.google.com/github/FaisalAbid11/Permutation-Entropy-vs-Modified-TOPSIS/blob/789260455ba34f695d8255722e018a7b00f04b39/AHP-MODIFIED-TOPSIS-EVALUATION/modfied_ahp_TOPSIS_20_30_50.ipynb)
- [AHP-TOPSIS](https://colab.research.google.com/github/FaisalAbid11/Permutation-Entropy-vs-Modified-TOPSIS/blob/b2cebc45da73259e06a60e42ede05dd186626616/AHP-TOPSIS-EVALUATION/ahp_TOPSIS_20_30_50.ipynb)
- [Permutaion-based-Entropy](https://colab.research.google.com/github/FaisalAbid11/Permutation-Entropy-vs-Modified-TOPSIS/blob/983bfaedc9d2c35a86497ad82c5ecd2dc252863f/Permutation-based-entropy/permutation_based_entropy.ipynb)
  
**Open from Colab:**
- Go to [Google Colab](https://colab.research.google.com/).
- Click on **"GitHub"** in the pop-up window.
- Paste the repository URL.
- Press Enter or click the search icon.
- Select the desired notebook (e.g., `modified_ahp_TOPSIS_20_30_50.ipynb`).
- It will launch in Colab, and you can run the cells directly in the browser.
  
***Note:You must be signed in with your Google account to use Google Colab.***
  
**2. Upload the required datasets when prompted, or mount your Google Drive.**

- **If you're loading datasets stored in your ***GitHub repository***, use the `raw` file URL:**
Click on the dataset file in the repository. Then, click the "Raw" button located at the top-right corner of the file preview. This will open the raw data in a new tab. From there, copy the URL from the address bar and paste it into the variable that will store the file path inside single or double quote.For Example, to open the distressed dataset:

```python
import pandas as pd
url = "'https://raw.githubusercontent.com/FaisalAbid11/Permutation-Entropy-vs-Modified-TOPSIS/refs/heads/main/AHP-MODIFIED-TOPSIS-EVALUATION/distressed.csv'"
df = pd.read_csv(url)
```
***Note: Change the URL name with the raw url of required dataset***

- **if you're uploading from your local system, first download it from datasets folder use the code to chose from local system:**
```python
from google.colab import files
uploaded = files.upload()
```
- **For larger or multiple files, it's better to upload file in google drive and mount it:**
```python
from google.colab import drive
drive.mount('/content/drive')
```
***Reminder: If you're not using Google Drive, you'll need to re-upload your files every time you reconnect to a Colab session***

**3. You can run all cell together by selecting Run all from Runtime option in menu bar or by pressing Ctrl+f9. But for better undersating run the notebook cells in order by pressing Ctrl+Enter for selected cell or by clicking the triangular icon/button on top left corner of the cell.**

### Option 2: Run Locally

#### 1. Clone the Repository
```bash
git clone https://github.com/FaisalAbid11/Permutation-Entropy-vs-Modified-TOPSIS.git
cd your-repository
```
#### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv env
source env/bin/activate    
```
#### 3. Install Required Packages
```bash
pip install numpy pandas scikit-learn matplotlib seaborn imblearn
```
#### 4. Run the Jupyter Notebooks
```bash
jupyter notebook
```
Then open the desired notebook (e.g., modified_topsis.ipynb) in your browser.

#### Including Datasets

- The dataset files (e.g., enthusiastic.csv, behavioral.csv etc) inside the datasets/ folder and so check and use correct path when using it.

# Process

## Modified AHP-TOPSIS

### Input Data:
A set of Employee Turnover Intention (TOI) features.

### Feature Selection Process:
The feature selection process involves integrating a modified version of the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) with the following methods:
- **Information Gain**: For evaluating the significance of features based on the amount of information they provide.
- **Recursive Feature Elimination (RFE)**: A method for eliminating less important features iteratively.
- **Select K-best**: A feature selection method that selects the top K features based on their statistical significance.

The objective of this process is to identify the most important features that affect employee turnover intention (TOI).

### Output:
This process will output a set of the most significant features associated with employee TOI.

### Categorization:
Once the features are selected, the **AHP-Modified TOPSIS** method is used to rank employees based on their turnover intention. The employees are then categorized into three groups using the **De-Pareto principle**

## Permutation-Based Entropy Method

### Input Data:
A set of Employee Turnover Intention (TOI) features.

### Feature Selection Process:
The feature selection method used here is a **permutation-based feature selection** technique. This method permutes feature values randomly to assess the importance of each feature in predicting employee TOI.

### Output:
This process will identify the key features that influence employee TOI.

### Categorization:
Once the important features are identified, an **Entropy-based weighted method**, integrated with TOPSIS, is employed for employee ranking. The employees are then categorized into three predefined groups

## Employee Categorization
The employee categorization is based on employee productivity and divided into three categories:
- **Enthusiastic**: The most productive employees.
- **Behavioral**: Employees with average productivity.
- **Distressed**: The least productive employees.
## Evaluation

The categorized datasets (*enthusiastic*, *behavioral*, and *distressed*) generated from both the **Permutation-Based Entropy + TOPSIS** and the **AHP-Modified TOPSIS + De-Pareto** methods were used to train and test a 5 Classifier models (**SVM,KNN,Random Forest,Logistic Regression,XGBoost**) using a **70/30 train-test split**.

To compare the effectiveness of the two feature selection pipelines, the following evaluation metrics were used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Matthews Correlation Coefficient (MCC)**
- **ROC-AUC**

This evaluation helped determine which method yielded better predictive performance in identifying employee turnover intention (TOI), ultimately highlighting the impact of the proposed **Modified TOPSIS** method. 

***To check the results , run the cells in order and evaluate the metrics for each method by running the notebooks***

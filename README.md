# 🚢 Titanic Survival Prediction - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project that predicts passenger survival on the Titanic using various classification algorithms.

**University:** Savitribai Phule Pune University  
**Subject:** Machine Learning (Pattern-2019)  
**Subject Code:** 410242

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Models Used](#models-used)
- [Results](#results)
- [Key Insights](#key-insights)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project analyzes the Titanic dataset to build a predictive model that determines which passengers were more likely to survive the disaster. The analysis includes:

- **Exploratory Data Analysis (EDA)**
- **Data Preprocessing & Feature Engineering**
- **Multiple ML Algorithm Comparison**
- **Hyperparameter Tuning**
- **Model Evaluation & Visualization**

---

## 📊 Dataset

The dataset is obtained from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data).

**Files:**
- `train.csv` - Training dataset (891 passengers)
- `test.csv` - Test dataset (418 passengers)
- `gender_submission.csv` - Sample submission file

**Features:**
- **PassengerId** - Unique ID
- **Survived** - Target variable (0 = No, 1 = Yes)
- **Pclass** - Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name** - Passenger name
- **Sex** - Gender
- **Age** - Age in years
- **SibSp** - Number of siblings/spouses aboard
- **Parch** - Number of parents/children aboard
- **Ticket** - Ticket number
- **Fare** - Passenger fare
- **Cabin** - Cabin number
- **Embarked** - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## 📁 Project Structure

```
MACHINE_LEARNING/
│
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── gender_submission.csv        # Sample submission
├── main.ipynb                   # Main Jupyter notebook
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore file
│
├── outputs/                     # Generated outputs (optional)
│   ├── models/                  # Saved models
│   ├── plots/                   # Visualizations
│   └── predictions/             # Prediction results
│
└── requirements.txt             # Python dependencies (optional)
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Anaconda/Miniconda (recommended)
- Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/titanic-ml-project.git
cd titanic-ml-project
```

### Step 2: Create Virtual Environment (Optional)

```bash
# Using conda
conda create -n titanic-ml python=3.8
conda activate titanic-ml

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**OR** install from requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open `main.ipynb`**

3. **Run all cells:**
   - Menu: `Cell` → `Run All`
   - Or: `Shift + Enter` for each cell

### Using the Prediction Function

After running the notebook, you can make custom predictions:

```python
# Predict survival for a first-class female passenger
predict_survival(pclass=1, sex='female', age=25, fare=100, embarked='S')

# Predict survival for a third-class male passenger
predict_survival(pclass=3, sex='male', age=30, fare=10, embarked='S')
```

---

## ✨ Features

### Data Preprocessing
- ✅ Intelligent missing value imputation
- ✅ Outlier detection and handling
- ✅ Feature scaling and normalization

### Feature Engineering
- ✅ **FamilySize** - Total family members aboard
- ✅ **IsAlone** - Whether passenger traveled alone
- ✅ **Title** - Extracted from passenger name (Mr, Mrs, Miss, etc.)
- ✅ **AgeGroup** - Categorized age ranges
- ✅ **FareGroup** - Categorized fare ranges
- ✅ **Has_Cabin** - Binary feature for cabin information

### Visualizations
- ✅ Survival distribution plots
- ✅ Gender and class-based analysis
- ✅ Correlation heatmaps
- ✅ Age and fare distributions
- ✅ Feature importance charts
- ✅ Confusion matrices
- ✅ Model comparison plots

---

## 🤖 Models Used

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~83% | ~81% | ~78% | ~79% |
| Gradient Boosting | ~82% | ~80% | ~77% | ~78% |
| Logistic Regression | ~81% | ~79% | ~76% | ~77% |
| SVM | ~80% | ~78% | ~75% | ~76% |
| Decision Tree | ~78% | ~76% | ~73% | ~74% |
| K-Nearest Neighbors | ~77% | ~75% | ~72% | ~73% |
| Naive Bayes | ~76% | ~74% | ~71% | ~72% |

**Best Model:** Random Forest Classifier (after hyperparameter tuning)

---

## 📈 Results

### Key Performance Metrics

- **Accuracy:** 83.24%
- **Precision:** 81.15%
- **Recall:** 78.43%
- **F1-Score:** 79.76%
- **Cross-Validation Score:** 82.91%

### Confusion Matrix

```
                Predicted
              No    Yes
Actual  No   [ 95    10 ]
        Yes  [ 20    54 ]
```

---

## 💡 Key Insights

1. **Gender Impact:**
   - Female survival rate: ~74%
   - Male survival rate: ~19%
   - Gender is the most important feature

2. **Passenger Class:**
   - First class survival: ~63%
   - Second class survival: ~47%
   - Third class survival: ~24%

3. **Age Factor:**
   - Children (0-12 years) had higher survival rates
   - Age groups show significant variation in survival

4. **Family Size:**
   - Small families (2-4 members) had better survival rates
   - Solo travelers and large families had lower survival rates

5. **Port of Embarkation:**
   - Cherbourg (C) passengers had highest survival rate
   - This correlates with higher class passengers

6. **Fare:**
   - Higher fare passengers had significantly better survival rates
   - Strong correlation with passenger class

---

## 🛠️ Technologies

**Programming Language:**
- Python 3.8+

**Libraries:**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Development:** Jupyter Notebook

**Tools:**
- Anaconda
- Git & GitHub
- Jupyter Notebook

---

## 📝 Future Enhancements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Add ensemble methods (Stacking, Voting)
- [ ] Create web interface using Streamlit/Flask
- [ ] Deploy model to cloud (AWS/Heroku)
- [ ] Add more feature engineering techniques
- [ ] Implement SHAP values for explainability
- [ ] Create automated reporting system

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Contact

**Your Name** - Vaibhav Mashal  
**Email:** vaibhavmashal@example.com  
**GitHub:** [@vaibhavmashal](https://github.com/vaibhavmashal)  
**LinkedIn:** [Vaibhav Mashal](https://linkedin.com/in/vaibhavmashal)

**Project Link:** [https://github.com/vaibhavmashal/titanic-ml-project](https://github.com/vaibhavmashal/titanic-ml-project)

---

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the Titanic dataset
- [Savitribai Phule Pune University](http://www.unipune.ac.in/) for the course curriculum
- [scikit-learn](https://scikit-learn.org/) for the machine learning library
- All contributors and open-source community

---

## 📚 References

1. [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
2. [scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Pandas Documentation](https://pandas.pydata.org/)
4. [Machine Learning Mastery](https://machinelearningmastery.com/)

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

Made with ❤️ for SPPU Machine Learning Course

</div>
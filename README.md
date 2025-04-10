
# 📧 Smart Email Classifier


The **Smart Email Classifier** is a system that classifies, evaluates, and refines email classifications using large language models (LLMs). The application utilizes a series of tasks and agents defined in **CrewAi**, an intelligent framework designed for managing tasks in AI pipelines.

### Features:
- Classify emails into categories such as **Spam**, **Urgent**, **Not Urgent**, and **To Read**.
- Evaluate the classification made by the model and provide feedback .
- Refine the classification if necessary based on evaluator feedback.
- Generate test cases for the email classifier and evaluate its performance through various metrics like Precision, Recall, and F1-Score.
  
The user interface is built with **Streamlit** for LLM-based classification and evaluation.

## Requirements

To run the project, you need the following dependencies:

- **Python 3.8+**
- **Streamlit**
- **CrewAi**
- **scikit-learn** 

You can run the app by directing to `src/ai` and running the following command:

```bash
streamlit run ui.py

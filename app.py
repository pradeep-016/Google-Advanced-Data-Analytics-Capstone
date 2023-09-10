import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#model = pickle.load('/model.pickle')
data = pd.read_csv('./HR_comma_sep.csv')
metrics = pd.read_csv('./metrics.csv')
st.title('''
         Google Advanced Data Analytics Capstone Project
         ''')
st.write("Churn Prediction and customer leaving the company analysis")
st.write("This capstone project is an opportunity to analyze a dataset and build predictive models that can provide insights to the Human Resources (HR) department of a large consulting firm.")
st.divider()
st.write('# Plan Stage')
st.write(data.head(10))
st.write(f'List of columns: ')
st.write([col for col in data.columns])
st.write(f'Describing dataset :')
st.write(data.describe())

st.write('''
         # Analyze Stage
         Reflect on these questions as you complete the analyze stage.
- What did you observe about the relationships between variables?
- What do you observe about the distributions in the data?
- What transformations did you make with your data? Why did you chose to make those decisions?
- What are some purposes of EDA before constructing a predictive model?
- What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
- Do you have any ethical considerations in this stage?
         ''')
st.write("Continuous EDA")
st.write("histogram showing distribution of 'number_project', comparing employees who stayed versus those who left")
st.image('./images/download1.png')

st.write("Detecting outliers on 'tenure' column")
st.image('./images/download3.png')
st.write("histogram showing distribution of 'number_project', comparing employees who stayed versus those who left")

st.image('./images/ltenure.png')
st.write("Scatter plot showing average_monthly_hours and satisfaction_level of customers")
st.image('./images/scatter.png')
st.write('Count of people stayed vs left by department')

st.image('./images/dept.png')
st.write("Null/ missing values")
st.write(data.isna().sum())
st.write(f'Duplicate values found : {data.duplicated().sum()}')
st.write('''
         # Construct Stage
- Determine which models are most appropriate
- Construct the model
- Confirm model assumptions
- Evaluate model results to determine how well your model fits the data
         ''')
st.write('Logistic regression is quite sensitive to outliers, it would be a good idea at this stage to remove the outliers in the tenure column that were identified earlier.')
st.write('''
         Logistic regression
Note that binomial logistic regression suits the task because it involves binary classification.
         ''')
st.write(metrics)
st.write('''
         The upper-left quadrant displays the number of true negatives. The upper-right quadrant displays the number of false positives. The bottom-left quadrant displays the number of false negatives. The bottom-right quadrant displays the number of true positives.
- True negatives: The number of people who did not leave that the model accurately predicted did not leave.

- False positives: The number of people who did not leave the model inaccurately predicted as leaving.

- False negatives: The number of people who left that the model inaccurately predicted did not leave

- True positives: The number of people who left the model accurately predicted as leaving

A perfect model would yield all true negatives and true positives, and no false negatives or false positives.
Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.
Check the class balance in the data. In other words, check the value counts in the left column. Since this is a binary classification task, the class balance informs the way you interpret accuracy metrics.
         ''')
st.image('./images/cm.png')

st.image('./images/feature.png')

st.write("# Execute Stage")
st.write('''
        - Interpret model performance and results
- Prepare results, visualizations, and actionable steps to share with stakeholders
         ''')

st.write('''
         #### Recall evaluation metrics
- AUC is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
- Precision measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
- Recall measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
- Accuracy measures the proportion of data points that are correctly classified.
- F1-score is an aggregation of precision and recall.
#### Summary of model results
##### Logistic Regression

The logistic regression model achieved precision of 80%, recall of 83%, f1-score of 80% (all weighted averages), and accuracy of 83%, on the test set.

##### Tree-based Machine Learning

After conducting feature engineering, the decision tree model achieved AUC of 93.8%, precision of 87.0%, recall of 90.4%, f1-score of 88.7%, and accuracy of 96.2%, on the test set. The random forest modestly outperformed the decision tree model.

#### Conclusion, Recommendations, Next Steps
The models and the feature importances extracted from the models confirm that employees at the company are overworked.

To retain employees, the following recommendations could be presented to the stakeholders:

- Cap the number of projects that employees can work on.
- Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
- Either reward employees for working longer hours, or don't require them to do so.
- If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
- Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
- High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort.
         ''')
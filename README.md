# Google Advanced Data Analytics
Google Advanced Data Analytics Capstone Project
This capstone project is an opportunity to analyze a dataset and build predictive models that can provide insights to the Human Resources (HR) department of a large consulting firm.

Access the app [here](https://app-advanced-data-analytics.streamlit.app/)

https://app-advanced-data-analytics.streamlit.app/


## TikTok Data Predictive Analysis using GridSearchCV and Random Forest
The main aim is to build a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently using machine learning techniques to predict on a binary outcome variable.

The **purpose** of this model is to mitigate misinformation in videos on the TikTok platform.

The **goal** of this model is to predict whether a TikTok video presents a "claim" or presents an "opinion".

This project use PACE framework (Plan, Analyze, Construct, Execute) and is divided to 3 parts: 

### Business need and modeling objective
 
TikTok users can report videos that they believe violate the platform's terms of service. Because there are millions of TikTok videos created and viewed every day, this means that many videos get reported—too many to be individually reviewed by a human moderator.

Analysis indicates that when authors do violate the terms of service, they're much more likely to be presenting a claim than an opinion. Therefore, it is useful to be able to determine which videos make claims and which videos are opinions.

TikTok wants to build a machine learning model to help identify claims and opinions. Videos that are labeled opinions will be less likely to go on to be reviewed by a human moderator. Videos that are labeled as claims will be further sorted by a downstream process to determine whether they should get prioritized for review. For example, perhaps videos that are classified as claims would then be ranked by how many times they were reported, then the top x% would be reviewed by a human each day.
### Modeling workflow and model selection process

Previous work with this data has revealed that there are ~20,000 videos in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:

- Split the data into train/validation/test sets (60/20/20)
- Fit models and tune hyperparameters on the training set
- Perform final model selection on the validation set
- Assess the champion model's performance on the test set
### Select an evaluation metric

To determine which evaluation metric might be best, consider how the model might be wrong. There are two possibilities for bad predictions:

- False positives: When the model predicts a video is a claim when in fact it is an opinion
- False negatives: When the model predicts a video is an opinion when in fact it is a claim
### Modeling design and target variable

The data dictionary shows that there is a column called claim_status. This is a binary value that indicates whether a video is a claim or an opinion. This will be the target variable. In other words, for each video, the model should predict whether the video is a claim or an opinion.

This is a classification task because the model is predicting a binary class.

![image](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)

The full explanation of code is available [here](https://www.kaggle.com/code/aleemaparakatta/tiktok-data-predictive-analysis-gridsearchcv/).

This model performs exceptionally well, with an average recall score of 0.995 across the five cross-validation folds. After checking the precision score to be sure the model is not classifying all samples as claims, it is clear that this model is making almost perfect classifications.
A machine learning model would greatly assist in the effort to present human moderators with videos that are most likely to be in violation of TikTok's terms of service.

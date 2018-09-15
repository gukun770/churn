#  Rideshare App Churn Prediction 

This casestudy is focused on rider last 30 days retention for a ride-sharing company X. The purpose of this notebook is to help understand what factors are the best predictors for retention, and offer suggestions to help Company X. [link to the notebook](http://nbviewer.jupyter.org/github/gukun770/churn/blob/master/Rideshare_App_Churn_Prediction.ipynb)

### Content
* [1. Data Description and EDA](#$$1.\-Data\-Description\-and\-EDA$$)
    * [1.1 Data Description](#1.1-Data-Description)
    * [1.2 EDA](#1.2-EDA)
* [2. Modeling](#$$2.\-Modeling$$)
    * [2.1 Model Training](#2.1-Model-Training)
    * [2.2 Feature Importance](#2.2-Feature-Importance)
    * [2.3 Feature Impact](#2.3-Feature-Impact)
* [3. User Resurrection](#$$3.\-User\-Resurrection$$)
    * [3.1 Cost Benefit Matrix](#3.1-Cost-Benefit-Matrix)
    * [3.2 Profit Curve](#3.2-Profit-Curve)


### Label: did a user churn? 
#### Definitions:

To help explore this question, a sample dataset of a cohort of users who signed up for an account in January 2014. **The data was pulled on July1, 2014**; A user is considered retained if they were “active” (i.e. took a trip)
in the preceding 30 days (from the day the data was pulled). In other words, a user is "active" if they have taken a trip since June 1, 2014. 

- **1 => CHURN**
    - The User **did not** use the ride sharing service in the last 30 days
- **0 => NO CHURN**
    - The user **did** use the ride sharing service in the last 30 days
    
-----------------

#### Problem Statement and background:


I use this data set to help understand **what factors are
the best predictors for retention**, and offer suggestions to help Company X. 

Here is a detailed description of the data:

- `city`: city this user signed up in phone: primary device for this user
- `signup_date`: date of account registration; in the form `YYYYMMDD`
- `last_trip_date`: the last time this user completed a trip; in the form `YYYYMMDD`
- `avg_dist`: the average distance (in miles) per trip taken in the first 30 days after signup
- `avg_rating_by_driver`: the rider’s average rating over all of their trips 
- `avg_rating_of_driver`: the rider’s average rating of their drivers over all of their trips 
- `surge_pct`: the percent of trips taken with surge multiplier > 1 
- `avg_surge`: The average surge multiplier over all of this user’s trips 
- `trips_in_first_30_days`: the number of trips this user took in the first 30 days after signing up 
- `luxury_car_user`: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise 
- `weekday_pct`: the percent of the user’s trips occurring during a weekday 

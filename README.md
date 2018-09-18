# Project Summary

The overall goal of this project was to look at models that could forecast disease-specific mortality over time. I initially envisioned building a more general model that would be able come up with forecasts for the most common diseases, but I eventually limited this first iteration of the project to forecasting mortality due to myocardial infarction, since cardiovascular disease is the number 1 cause of mortality in the United States (and has been for the past 80 years!)

## Project design & tools
The way I designed my project was to pick the mortality rate from myocardial infarction over time as a target variable, and then also extracting other variable time series that I thought were plausible predictors that could influence the target variable, e.g. smoking status, prevalence of diabetes, prevalence of obesity, etc. Several forecasting methods use only past values of the target variable to predict/forecast future values, but I was curious about whether or not adding in these other predictors (also called covariates or exogenous variables) would affect or even improve the forecasts.

## Data
For my target variable, I queried the CDC WONDER database, which is able to provide fairly granular mortality statistics. I was able to query the database to get the monthly number of deaths due to myocardial infarction per state, for the years of 1999 through 2016; this yielded 216 observations for 50 different time series, one for each state. I then used these time series to calculate the number of deaths due to MI per 100,000 people in order to make the rates comparable across states, and this ended up being my target variable.

For my exogenous variables, I used data from the Behavioral Risk Factor Surveillance System ([BRFSS](https://www.cdc.gov/brfss/about/index.htm)), which is an annual telephone survey run by the CDC that collects data from US residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. The BRFSS started in 1984 with 15 states participating, but now every state and the District of Columbia participate in the BRFSS, which is a fairly comprehensive survey; there are roughly 300 variable/question pairs in the codebooks for recent surveys! I looked through the codebooks and ended up picking the following demographic, socioeconomic, and medical variables to extract because I thought that these variables could plausibly affect mortality due from heart attack:

### Demographic/socioeconomic:
* Gender
* Income
* General health
* Any exercise in past month
* Number of days in past month with poor mental health
* Unable to afford medical care in past year

### Medical
* High cholesterol
* Hypertension
* Diabetes
* Obesity

Extracting this data was a big challenge, because each year had a codebook as a pdf file, and I had to parse the text of each codebook to create a dictionary mapping each variable to each question. But, different variable names would change every few years, and not in a consistent way, which meant that I had to construct groups of synonyms for each relevant variable in order to extract the data. But I was eventually able to get the data I needed in order to calculate the statewide prevalence for the response of interest for the appropriate variables (e.g. diabetes prevalence, obesity prevalence, etc); for some of the other variables, I calculated a different quantity, e.g. for income I calculated the median income, for number of days in the past month with poor mental health, this ranged from 0 to 30, etc.

## Algorithms/Models
I initially started modeling with SARIMA, which is a model that uses only the target variable itself to make predictions. I did a brute force search over hyperparameter values of p, d, and q and also P, D, and Q (seasonal parameters, with m = 12) ranging from 0 to 1, and I trained models on 10 years of data, and ranked the models based on the test RMSE to forecast the next 5 years. For California, the model that performed the best was SARIMA(1, 1, 0)x(1, 0, 1, 12). However, this model is quite sensitive to the time point when it's asked to start forecasting; when I used those parameters and trained it on 9 years of data or 11 years of data, the 5 year forecasts that it generated were quite different (and quite a lot worse!) than the best forecast that it had produced when trained on 10 years.

I also wanted to see if these 'best' hyperparameter values for California were generalizable to other states; for some of the states, these hyperparameter values did create models that forecasted well, but for the most part, they did not.

I then moved onto incorporating my exogenous variables into a forecasting model by using SARIMAX. I again performed a brute force search over the same set of hyperparameters, again training on 10 years of data and ranking models on their forecasts for the next 5 years. The 'best' forecasting model actually had different pdq/PDQ values (it was a SARIMAX(0,0,0)x(1,0,1,12)), and also the forecast was actually *less* accurate with the exogenous variables compared to the SARIMA model without exogenous variables!

I was also interested in figuring out which feature was the most important to the accuracy of the forecast, and so the way I evaluated that was shuffled each predictor and calculated how much the RMSE for the 5 year forecast increased. I shuffled each exogenous variable a hundred times and took the mean RMSE, and at the end of that I got this list ranked from most important to least important. I got a ranked list of variables, with health care coverage being the most important variable to the accuracy of the SARIMAX model, but I also wanted to see if this model would give plausible forecasts for the full range of the exogenous variables.

I tested this by seeing what would happen if we saturated or 'maxed out' each predictor. That is, what would happen to the forecast if we told the fitted model that for the next 5 years, 100% of people have diabetes? Or 100% of people won’t do any exercise? What I found was that for these two variables, the model did forecast an increased number of cardiac deaths per month for either of these variables at their max values, which is potentially plausible, but other variables 'maxing out' did not give plausible forecasts. For example, when I told the model that 100% of people in California have high cholesterol for the 5 years that I want it to forecast, the model said that about -50 people per 100K per month would die from heart attack, so clearly the model is trying to encourage us to go eat more McDonalds!

## Further work
I did some experimenting with other algorithms e.g. gradient boosted trees, random forests, etc. to see how well they would forecast, but they didn't work very well. I think that it would be interesting to use neural nets - specifically RNNs or LSTMs since these networks can 'remember' past values, but I think that this project was also limited by the time frame of this data. Specifically, none of these models or algorithms can answer the question, what is the appropriate lag for each exogenous variable to influence the target? California has had an increasing prevalence of obesity and diabetes over the past 16 years but a *decreasing* mortality due to heart attack over that same time period, but it would be incorrect to conclude that higher obesity and diabetes rates means lower heart attack mortality! It's entirely possible that the lag time for these higher obesity and diabetes rates to cause higher heart attack mortality rates is on the order of years, and we just don't have enough data yet to look at it.

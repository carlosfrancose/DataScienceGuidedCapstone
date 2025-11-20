# Guided Capstone Project Report
## Overview
### Context
Big Mountain Resort, a ski resort in Montana, currently prices their tickets based on market averages. While this serves as a good baseline, without a data-driven understanding of resort pricing, they risk leaving money on the table. To address this problem, Big Mountain Ski Resort has collected a rich dataset on the various ski resorts across the United States. This dataset contains key insights on the various facilities and features each resort offers, giving us a great opportunity to identify potential correlations with ticket prices. The hope is that a more refined look at these relationships can help evaluate Big Mountain's pricing strategy, and determine if operational changes can be made to increase revenue or reduce maintenance costs.

### (*SMART*) Problem Statement
How can Big Mountain Resort accurately predict ski lift ticket prices based on a resort's physical characteristics and operational facilities, so that they can make an informed decision on increasing their own ticket prices or reducing facilities for the upcoming season?

### Objective
Utilizing the dataset collected by Big Mountain Resort, we'll develop a regression model that takes various resort facilities and targets (weekend) ticket prices as a prediction.

## Data
### Data Wrangling
The raw dataset provided by Big Mountain Resort has a total of 330 entries, with 27 different features, including our target feature of ticket price.

|#|Column|Non-Null Count|Dtype|
|---|---|---|---|
|0|Name|330|object|
|1|Region|330|object|
|2|state|330|object|
|3|summit_elev|330|int64|
|4|vertical_drop|330|int64|
|5|base_elev|330|int64|
|6|trams|330|int64|
|7|fastEight|164|float64|
|8|fastSixes|330|int64|
|9|fastQuads|330|int64|
|10|quad|330|int64|
|11|triple|330|int64|
|12|double|330|int64|
|13|surface|330|int64|
|14|total_chairs|330|int64|
|15|Runs|326|float64|
|16|TerrainParks|279|float64|
|17|LongestRun_mi|325|float64|
|18|SkiableTerrain_ac|327|float64|
|19|Snow Making_ac|284|float64|
|20|daysOpenLastYear|279|float64|
|21|yearsOpen|329|float64|
|22|averageSnowfall|316|float64|
|23|AdultWeekday|276|float64|
|24|AdultWeekend|279|float64|
|25|projectedDaysOpen|283|float64|
|26|NightSkiing_ac|187|float64|

Initial data cleaning efforts resulted in dropping 2 columns, and 53 rows. The first column, *fastEight*, was nearly half null values, while the majority of the non-null values had an entry of zero. Thus this column was determined to be too sparce to include. The second column, *AdultWeekday*, was dropped as a result of determining *AdultWeekend* to be a better target. This conclusion came after evaluating the differences in the weekday and weekend ticket prices for all entries and finding that for the vast majority of cases both ticket prices were the same, with resorts on the lower end of pricing being the only ones to show any significant difference. Additionally, *AdultWeekend* contained slightly fewer missing entries than *AdultWeekday*.

[INSERT FIGURE 1]

The vast majority of rows removed were removed simply for having a null *AdultWeekend* entry. One additional row was removed as it was determined that based on *yearsOpen* that it was most likely an erroneous entry. The resulting dataset after wrangling has 25 columns, 277 rows, and is primed for further exploratory analysis. 

### EDA
#### State Significance
Following the brief data cleaning, we began our exploratory data analysis with a slight detour in mind. What influence do states have in ticket pricing? We start this rabbit hole by loading state data from Wikipedia that we pulled during the data wrangling phase, using it to construct ratio features that will help us further evaluate our potential relationships. The custom features, stored in the dataframe `state_summary`, are as follows:
- *resorts_per_state*
- *resorts_per_100kcapita*
- *resorts_per_100ksq_mile*
- *state_total_skiable_area_ac*
- *state_total_days_open*
- *state_total_terrain_parks* 
- *state_total_nightskiing_ac*  

The histograms for both for resorts per state's 100k capita and 100k square miles show pretty strongly right skewed histograms:

[INSERT FIGURE 2 AND 3]

While the histogram for distribution of state average prices appears to be somewhat bimodal:

[INSERT FIGURE 4]

This seems to imply slightly differentiable classes of states. The first two figures on resort density seem to be indicative of some 6 or so states that contain well above average amounts of ski resorts, both in population density and land. The mean distribution seems to also show a split in ticket pricing, this time with two peaks implying a significant number of states command a higher average price, irregardless of land mass or population density. Were I conducting this EDA on my own, I would be interested in exploring this further, identifying which states seem to be the outliers for each of these scenarios. However, the guided notebook seems to go in a slightly different direction.

Per the guided notebook's instructions, we conducted a primary component analysis on the state summary data frame, which frankly yielded little conclusive evidence on our initial question of state importance. There may be some states that could arguably be considered differentiated significantly enough from others to be grouped separately. Were I or Big Mountain Resort interested in delving further down this rabbit hole, I would suggest cross referencing the PCA outliers with the outliers of the earlier distributions, and see if any patterns emerge. However, this is ultimately where we end this exploration. Relatively inconclusive, and ultimately not of any use to our final model.

#### Target Feature & Correlations
I've read online that it's recommended to start EDA with an in depth target feature analysis. While the guided notebook opted for a state analysis instead, we do get around to creating some real informative visualizations with a correlation heatmap and a large multi-scatterplot figure with each numerical feature.

[INSERT FIGURE 5]

This heatmap provides arguably the most significant insights from the entire EDA notebook, as the features we see with strong correlation to the target feature *AdultWeekend* here prove to be the same features we select during our modeling process later. Some of these features include:
- *vertical_drop*
- *Runs*
- *Snow Making_ac*
- *fastQuads*

Interestingly, *fastQuads* stood out for having a much stronger correlation than one might initially assume, even stronger than the similar feature *total_chairs*. We later explored this comparison briefly near the end of the notebook by comparing both features in ratios with *Runs* and *SkiableTerrain_ac*.

[INSERT FIGURE 7]

My initial interpretation of these graphs led me to believe that *total_chairs* provided a more realistic spread of how chairlifts affects ticket prices. However, we would later find in the **preprocessing** phase that again, *fastQuads* has noticeably more significance in ticket price prediction than *total_chairs*. Now looking at these scatterplots in hindsight, my hypothesis is that *fastQuads* shows more deterministic split between resorts with no fast quad lifts and resorts with fast quad lifts, both in at noticeably differentiable ticket price ranges.

To further evaluate ticket pricing spread across all of our numerical features, we created a scatterplot for each feature:

[INSERT FIGURE 6]

The majority of these relationships appear to be nonlinear, with exception to some of the features with high correlation such as *vertical_drop*, *Runs*, *Snow Making_ac*, and, again, even *fastQuads*, each showing clear positive trends with ticket price. Some features such as  *resorts_per_100kcapita* and *resorts_per_100ksq_mile* interestingly show a bit of a horizontal convergence, suggesting that as a state's number of resorts increase, price becomes more deterministic.

#### Key Takeaways
- Ticket prices seem to primarily correlate with physical ski resort attributes, as well as a few esteemed operational features, i.e. vertical drop, skiiable terrain size, number of runs, snowmaking capacity, and high-speed lifts.
- Many lift-related and elevation-related features are highly correlated, signaling that likely only one of each of these groupings will be used in later modeling.
- It's unlikely that state-level features will provide any additional predictability to the model.
- The presence of nonlinear relationships suggested that tree-based models may perform better than strictly linear ones.

### Preprocessing
After a pretty longwinded EDA process, we moved onto the step of preprocessing, in the '04_preprocessing_and_training' notebook. The vast majority of the work done within this notebook pertains to the model training process, which we will detail more of in the ***Model*** section.

However, in terms of data preprocessing, the primary steps taken were:
- Dropping categorical features.
- Dropping the Big Mountain Resort entry from the data set.
- Splitting our data into a 70/30 train-test split.
- Substituting null values with their median.

The columns *Name*, *state*, and *Region* were dropped from the set prior to our model training. These features were non-numeric and thus would have to be encoded if we wanted to move forward with maintaining them. However, based on our findings from the EDA we preformed earlier, these features were not expected to provide meaningful contribution to prediction. Removing them allowed the models to focus on quantitative resort characteristics directly tied to pricing.

A 70/30 train-test split was applied to create a reliable method of validation for after training. We used 5-fold cross validation and scikit-learn Pipelines to prevent any data leakage while training. We also enforced null value substitutions through the Pipelines' `SimpleImputer`, ultimately settling on using the median strategy after performing a hyperparameter test with GridSearchCV.

It's worth noting that scaling the data is typically part of preprocessing, and while we did scale our data for the linear regression model we would test, ultimately our hyperparameter testing revealed that our final random forest model performed better with no scaling when compared to using `StandardScaler()`.

## Model
### Baseline Model & Evaluation Metrics
A DummyRegressor using the mean of the training labels was included to establish a baseline level of performance. This model produced an MAE of roughly **$19.14**, helping ground our expectations for what constituted meaningful improvement.

This baseline model served as a good introduction to evaluation metrics, specifically R squared, mean squared error (MSE), and mean absolute error (MAE). We ultimately settled on using MAE as our primary evaluation metric because unlike R squared, it offers pretty direct business interpretability by using the same units as the target feature, which is dollars. Additionally, MSE is particularly punishing for outliers, which isn't great for our particular use case of evaluating various ski resorts from all over the country.

### Candidates & Hyperparameters
Building off the baseline model, two different models were optimized and tested: Linear regression and random forest regression. Both models were wrapped in pipelines to make simplify the imputing, scaling, and fitting process. We also fed both pipelines into GridSearchCV to tune the models' hyperparameters with 5-fold cross-validation.

For the linear regression pipeline, we primarily focused on feature selection, utilizing `selectkbest__k` to determine the best subset of predictive features. GridSearchCV determined that the best performing linear model used **8 features**, which were: *vertical_drop*, *Snow Making_ac*, *total_chairs*, *fastQuads*, *Runs*, *LongestRun_mi*, *trams*, and *SkiableTerrain_ac*. It's worth noting the overlap between these features and the features identified with high correlation during EDA. However, despite this accurate selection of features, the optimized linear regression model still yielded a MAE of **$11.79**. This result is significantly better than the baseline, but still shows the limitations of using linearity to predict a mostly nonlinear dataset.

The random forest pipeline took a different approach. Tree-based models naturally handle nonlinear relationships and are robust to correlated features, so all available numeric features were retained. GridSearchCV tested number of estimators, mean and median imputation strategies, and the presence or absence of scaling. Ultimately the best regressor selected had a configuration with 69 trees, median imputation, and no scaling something that I was surprised to learn is typical behavior for tree-based models. This optimized random forest achieved a test-set MAE of **$9.54**, outperforming both the baseline and the linear model.

### Model Selection
Between the two optimized models, the **Random Forest Regressor** demonstrated the strongest overall performance, achieving the lowest cross-validated MAE and showing greater consistency on the test set. It also placed feature importance rankings closely aligned with the high correlation features we saw earlier in EDA, with *fastQuads*, *Runs*, *Snow Making_ac*, and *vertical_drop* being far and beyond the top four weighted features. For these reasons, the Random Forest model was selected for final training and scenario analysis.

### Scenario Modeling
To translate the model predictions into actionable measures, we created a custom scenario modeling function that applies the finalized Random Forest pipeline to a modified version of Big Mountain Resort, with new specified parameters (such as altering vertical drop or adding snowmaking acreage). This function returns the difference between the hypothetical scenario's predicted price and the current predicted price, allowing us to isolate the effect of individual operational changes while keeping all other features constant, creating an ideal playground for analysts.


## Results
### Model Prediction
Using the finalized Random Forest Regressor fit on the full dataset, Big Mountain Resort’s predicted Adult Weekend ticket price was **$95.87**. Compared to the resort’s current price of **$81.00**, this suggests that Big Mountain is underpriced relative to comparable U.S. ski resorts. Even accounting for the model’s MAE (~**$10.39**), the resort should be able to support at least a modest increase of **$4.48** without exceeding reasonable market expectations.

### Scenario Exploration
Using our custom scenario modeling function, we evaluated four hypothetical business scenarios by modifying Big Mountain’s feature values and measuring the change in predicted ticket price relative to the baseline **$95.87** prediction.

#### Scenario 1
Closing down the resort's least used runs, from -1 to -10.
##### Results:
|Runs Removed|Price Change ($)|
|---|---|
|1|0.00|
|2|–0.41|
|3|–0.67|
|4|–0.67|
|5|–0.67|
|6|–1.26|
|7|–1.26|
|8|–1.26|
|9|–1.71|
|10|–1.81|
Strangely, the model predicts that removing 1 run should have no effect on ticket price. It's worth considering that the nature of tree-based models such as Random Forests is to make "piecewise" predictions, meaning that the difference of 1 run likely just didn't create a significant enough difference to the model for it to change its prediction. It's safe to assume that any removal of a run would cause some level of price decrease, but in order to get a more accurate prediction on the removal of one run (or more), we likely need a bit more data granularity.

#### Scenario 2
Add one run, increase vertical drop by 150 ft, and add one chairlift.
##### Results:
Predicted price change: **+$1.99**
Predicted ARR change: **+$3,474,637.68**

This scenario likely yields a noticeable positive change in ticket pricing. Increasing the vertical drop, number of runs, and adding an additional chairlift directly improves on the resort's throughput and overall service availability.

#### Scenario 3
Same as Scenario 2, but also add 2 acres of snowmaking.
##### Results:
Predicted price change: **+$1.99**
Predicted ARR change: **+$3,474,637.68**

The results are identical to Scenario 2, implying that the additional snowmaking is not significant enough to increase pricing beyond the changes made in Scenario 2. This may suggest that small increases in snowmaking have insignificant returns unless they are accompanied by a larger-scale expansion.

#### Scenario 4
Increase longest run by 0.2 miles and add 4 acres of snowmaking.
##### Results:
Predicted price change: **+$0.00**

This scenario yields no predictable change in ticket price, suggesting that extending the longest-run alone does not have a significant influence on consumer's willingness to pay.

### Business Insights
- Big Mountain Resort is likely undervaluing their ticket prices, with our model predicting tickets within the $85–$106 range.
- Improvements tied to vertical drop, runs, and lift capacity deliver the strongest pricing impact, consistent with the model’s top four features (*fastQuads*, *Runs*, *Snow Making_ac*, *vertical_drop*).
- Scenario 2 offers the clearest revenue upside (+$1.99), with relatively moderate operational investment.
- Snowmaking-only expansions like Scenario 3 & 4 seemingly provide little to no measurable pricing benefit.
- Scenario 1 is likely to decrease revenue, although further evaluation is worth considering if more data becomes available.

Overall, Big Mountain should raise its base price modestly and consider targeted infrastructure improvements that increase vertical and lift throughput similar to Scenario 2.
### Assumptions
The validity of the model's output is dependent on several assumptions:
- Ticket pricing is driven primarily by physical and operational features, and not external market forces such as regional demand or brand awareness.
- The dataset is a comprehensive, accurate, and up to date representation of national ski resorts.
- Scenario modeling includes all changed variables while all other variables, including those unmodeled, remain the same or unaffected.

Additionally, the model currently does not account for potentially critical sources of data such as annual attendance, customer satisfaction, or operational costs. Any of these sources of data could significantly strengthen of our model's predictions, and the model's inability to account for these in its current state must be considered.

## Next Steps
### Deployment
To make our Random Forest model more accessible to Big Mountain Resort's analytics team, we could integrate our pipeline into their internal applications or dashboards by deploying it as serialized object served through an internal API, or schedule it as a batch process that feeds predictions into their existing data warehouse or reporting tools.

### Future Exploration
The scenario modeling function we developed offers a flexible tool for future analytical endeavors. Incorporating this function into our deployment would allow for analysts to explore different operational cost-benefit tradeoffs. If we manage to include richer data sources, like annual attendance, regional demand, or even operational costs, we can consolidate a dependable, robust decision-support system with far more comprehensive analysis for Big Mountain Resort's future projects.

## Conclusion
### Summary 
This project was a comprehensive application of the ML lifecycle to evaluate Big Mountain Resort's ticket pricing strategy, deriving a baseline model built to predict any American ski resort's ski lift ticket prices based on the resorts physical attributes and operational amenities. Additionally, we were able to establish a rudimentary function for applying hypothetical parameter changes, reflecting potential changes in ticket pricing that may occur as a result of Big Mountain's renovations.

Extensive EDA and model behavior allows us to conclude that the park's number of runs, vertical drop, snow making acreage, and number of available ski lifts (particularly fast quads) all have some significant contribution to ticket pricing. Furthermore, our model predicts that even with all current amenities unchanged, Big Mountain Resort is likely underpricing its current pricing by at least **$4.48**.

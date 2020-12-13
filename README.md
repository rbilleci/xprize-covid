SATURDAY:

* github
* time plan
* new kb cable?
* PREPARE UPWORK
* clean aws account

------------------------------------------------------------------------------------------------
SUNDAY:

* plot our work for a country to make sure the data manipulation is ok COMMIT UPWORK get it running on the beast
* data might change for countries, so we must AT MINIMUM find a way to hash them!... anything where data can be added
* with so many classes, we may need to hash them

            * filter out unknown columns on initial read
            * filter out unknown columns in final output
            * sanity check at each stage on the columns we are reading or writing!!!
            * better scaling of fields with order-of-magnitude values
            * check for missing days, fill them in if they are gone?
            * sanity checks at every stage
* extract some key properties out to a single file, like training ratio splits
* double check the training ratio splits
* work on our basic type system/naming scheme, simple helper tools to map logical name to real name

------------------------------------------------------------------------------------------------

* fault tolerance: inject new columns, add new columns to test reliability

# do we need to close/unlink files????
------------------------------------------------------------------------------------------------

# B

    * k-fold validation
    * how should we handle the input
    * move to 7 day rmse?
    * think better about the split
    * see how evaluation is done in the examples, to compare

# experiments

# REVIEW https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_2019
------------------------------------------------------------------------------------------------        

# TEST IN ENVIRONMETN CONSTRAINTS

    Following are the specifications of each evaluation sandbox. Teams should remember that their
    predictor must return a prediction in less than 1 hour for up to 180-days of prediction for up to 300
    regions when it is called.
        ● OS: Ubuntu Bionic 18.04
        ● CPU: 2
        ● RAM: 8 Gb

        
------------------------------------------------------------------------------------------------     

# IMPORTANT

    * adapting the model predictions using the data we get each day, measuring by how far we might be off

# NEXT

    * bootstrap it to actually work when deployed!!!
    * look into slack channel, to get hints
    * look into github and gitlab for other projects doing something similar, to borrow
    * compare results with reference implementation!

------------------------------------------------------------------------------------------------

# platform

    * framework for evaluating tons of different alogorhtms and approaches automagically

------------------------------------------------------------------------------------------------

* is working day vs is holiday -> it is important to identify dates that might be holidays or any past or future dates
  that might be holidays
* populations

# OTHER

* ensemble approach

* better to evaluate against N day average instead????
* how long each intervention has been in-place
* PEP8
* create sample datasets based on day/country/.... population double check values for all monetary data!
  population density
  !!!!!!!!!!! * what can we do about the cases/deaths in the data? do we need to remove them? !!!!!!!!!!!!!!!!
  !!!!! AWESOME SOURCE OF INFO !!!
  https://www.worldometers.info/coronavirus/
  https://www.worldometers.info/population/
  -> reorganize into functions and pipelines, pluggable, independent (maybe don't drop columns until the end)
  -> investigate why some countries have no continent, and how to fix or map???? -> some of the intervention stuff is
  log. we might want to descale it!
* more intelligent scaling mechanisms... means log, lin....
* review: https://covid19.eng.ox.ac.uk/data_schema.html
* look again at dense embeddings? -> maybe make it optional
* fully
  understand: https://aws.amazon.com/blogs/machine-learning/maximizing-nlp-model-performance-with-automatic-model-tuning-in-amazon-sagemaker/
* what to measure? - val acc? - f1 score?* see evaluation based on validation accuracy!
* check
    - https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-produces-better-models-faster/
    - https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-becomes-more-efficient-with-warm-start-of-hyperparameter-tuning-jobs/
* see ReverseLogarithmic
  here: https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-now-supports-random-search-and-hyperparameter-scaling/
* get a modern hyper parameter example: https://sagemaker.readthedocs.io/en/stable/v2.html
* preprocessing with scikit learn: https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
* analysis on which features have the most impact
* Good
  Examples https://github.com/aws/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/xgboost_random_log/hpo_xgboost_random_log.ipynb

# projecting out the number of working days in the future

# project the number of working days in the past

* school seasons (uni, middle, etc...)

# distance? coordidates? geo-data?

* directions/change
* moving averages
* special holidays?
* election days / election year (varies by country)
* number of holidays in next N days
* number of holidays in last N days
* disease column
* better handling of workig days for bad countries!!!!
* compare with and without OHE for ordinal values
* see http://www.city-data.com/
* performance
* use smaller data types, instead of float 64, especiall since most things are booleans!!!!
* is the data balanced?
* what if we treat each countries data independently, to get more data, and then blend results?
* working day calculation is too slow!

# Stretch Goals

    * cycles of other calendars, ramadan, ...
    * blend results with different models?
    * adjust prediction as we go on?
    * have two versions of the dataset, one OHE, and the other not https://stats.stackexchange.com/questions/95212/improve-classification-with-many-categorical-variables
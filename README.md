
------------------------------------------------------------------------------------------------
SUNDAY:
* speed up initialization
MUST READ: https://machinelearningmastery.com/improve-deep-learning-performance/
* get some kind of batch running overnight  
* bootstrap to work in actual environment
* review: https://www.datacamp.com/community/tutorials/deep-learning-python  
# review tutorials: https://www.tensorflow.org/tutorials/keras/regressionx
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
* factor out country generation to a separate file
* get a notebook to explore the data
* evaluate not NN algorithms
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
* REVIEW https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_2019
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

------------------------------------------------------------------------------------------------


# OTHER

* ensemble approach
* PEP8
* create sample datasets based on day/country/.... population double check values for all monetary data!
  -> investigate why some countries have no continent, and how to fix or map???? -> some of the intervention stuff is
* review: https://covid19.eng.ox.ac.uk/data_schema.html
* look again at dense embeddings? -> maybe make it optional
* fully
* check
    - https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-produces-better-models-faster/
    - https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-becomes-more-efficient-with-warm-start-of-hyperparameter-tuning-jobs/
* see ReverseLogarithmic
  here: https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-now-supports-random-search-and-hyperparameter-scaling/
* get a modern hyper parameter example: https://sagemaker.readthedocs.io/en/stable/v2.html
* preprocessing with scikit learn: https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
* analysis on which features have the most impact
  Examples https://github.com/aws/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/xgboost_random_log/hpo_xgboost_random_log.ipynb
* school seasons (uni, middle, etc...)

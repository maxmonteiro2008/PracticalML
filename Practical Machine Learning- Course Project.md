# Practical Machine Learning- Course Project
Authot: Maxwell
Date:08/11/2020

## Executive summary
The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

## Loading Data
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

library(caret)
library(rpart)
library(doParallel)
training<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                      sep = ",", na.strings = c("", "NA", "#DIV/0!"), strip.white = TRUE, stringsAsFactors = FALSE)

testing<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                    sep = ",", na.string=c("", "NA", "#DIV/0!"), strip.white = TRUE, stringsAsFactors = FALSE)
## Cleaning Data
In order to clean the data I proceeded the all NA columns removing, and the near zero covariantes removing

training$classe<-factor(training$classe)

training <- training[,8:length(colnames(training))]
testing <- testing[ ,8:length(colnames(testing))]

training<- training[ , colSums(is.na(training)) == 0] 
testing <- testing[, colSums(is.na(testing)) == 0] 

nzv<- nearZeroVar(training,saveMetrics=TRUE)
training<- training[,nzv$nzv==FALSE]
Cross Validation Slicing
Here i sampled 80% of the training data for a training set and 20% for a cross validation set

library(caret)
library(rpart)
intraining <- createDataPartition(training$classe, p=0.80, list=F)
trainingset <- training[intraining, ]
crossvalidationset <- training[-intraining, ]

## Model Fitting
I decided to fit a random forest model optmized by a k-fold cross validation

set.seed(825)

fitControl <- trainControl(## 10-fold CV
                           method = "cv",
                           number = 5,
                            allowParallel = TRUE)


gbmFit1 <- train(classe ~ ., data = trainingset, 
                 method = "rf", 
                 trControl = fitControl,
                  metric='Accuracy',
                 verbose = FALSE)



gbmFit1
###  Random Forest 
 
15699 samples
52 predictor
5 classes: 'A', 'B', 'C', 'D', 'E' 
 
 No pre-processing
 Resampling: Cross-Validated (5 fold) 
 Summary of sample sizes: 12558, 12560, 12559, 12560, 12559 
Resampling results across tuning parameters:
 
   mtry  Accuracy   Kappa    
    2    0.9926747  0.9907334
   27    0.9925474  0.9905728
   52    0.9884067  0.9853357
 
 Accuracy was used to select the optimal model using the largest value.
 The final value used for the model was mtry = 2.
As can be seen above, The model reached 99.2% of accuracy, using 2 predictors.

## Testing the model
Here we test the model , checking its accuracy

predtrainingset <- predict(gbmFit1, trainingset)
print(confusionMatrix(predtrainingset, trainingset$classe))
### Confusion Matrix and Statistics

           Reference
 Prediction    A    B    C    D    E
          A 4464    5    0    0    0
          B    0 3032    4    0    0
          C    0    1 2734    9    0
          D    0    0    0 2564    0
          E    0    0    0    0 2886
 
### Overall Statistics
                                          
                Accuracy : 0.9988          
                  95% CI : (0.9981, 0.9993)
     No Information Rate : 0.2843          
     P-Value [Acc > NIR] : < 2.2e-16       
                                           
                   Kappa : 0.9985          
                                           
  Mcnemar's Test P-Value : NA              
 
Statistics by Class:
 
                      Class: A Class: B Class: C Class: D Class: E
 Sensitivity            1.0000   0.9980   0.9985   0.9965   1.0000
 Specificity            0.9996   0.9997   0.9992   1.0000   1.0000
 Pos Pred Value         0.9989   0.9987   0.9964   1.0000   1.0000
 Neg Pred Value         1.0000   0.9995   0.9997   0.9993   1.0000
 Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
 Detection Rate         0.2843   0.1931   0.1742   0.1633   0.1838
 Detection Prevalence   0.2847   0.1934   0.1748   0.1633   0.1838
 Balanced Accuracy      0.9998   0.9989   0.9989   0.9983   1.0000

** plot(gbmFit1)**
 The above results, show 100% of accuracy in the training set( confirming model’s consistency ).Further more, the graph shows how the accuracy decreases as more predictors are included.

Predicting the results
library(caret)
ptest <- predict(gbmFit1, testing)
result<- table(ptest,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))


knitr::kable(result, caption = "Prediction",espace = TRUE)
Prediction
  1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20
A	0	1	0	1	1	0	0	0	1	1	0	0	0	1	0	0	1	0	0	0
B	1	0	1	0	0	0	0	1	0	0	1	0	1	0	0	0	0	1	1	1
C	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0
D	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0
E	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	1	0	0	0	0

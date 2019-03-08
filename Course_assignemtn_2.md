Read data and set seed
----------------------

``` r
data <- read.csv("trainingdata.csv", na.strings = c("NA","", "#DIV/0!"))
set.seed(1234)
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(rpart)
library(rpart.plot)
library(lattice)
```

get rid of useless rows that I don't want to use in my model
------------------------------------------------------------

``` r
data2 <- data[,-c(1:7)]
# Get rid of rows with NA
data3 <- data2[,colSums(is.na(data2)) == 0]
```

Split data so I can use 75% to train and 25% to test my prediction
------------------------------------------------------------------

``` r
data3 <- data2[,colSums(is.na(data2)) == 0]
```

Cross Validation - Split data so I can use 75% to train and 25% to test my prediction
-------------------------------------------------------------------------------------

``` r
trainingdata <- createDataPartition(y=data3$classe, p = 0.75, list = FALSE)        
train <- data3[trainingdata, ]
testtrain <- data3[-trainingdata, ]
```

I want to check how often each classes come up
----------------------------------------------

``` r
plot(train$classe, main = "Plot of Classe Variable in Training Data", xlab = "Classe", ylab = "Frequency", col = "blue")
```

![](Course_assignemtn_2_files/figure-markdown_github/unnamed-chunk-5-1.png) I can see that A is a little more frequent, but they are spread out pretty evenly \# Model 1 - Decision Trees

``` r
model1 <- rpart(classe ~ ., data=train, method="class")
prediction1 <- predict(model1, testtrain, type = "class")

# Plot the prediction
rattle::fancyRpartPlot(model1, main = "prediction 1 Classification Tree")
```

    ## Warning: labs do not fit even at cex 0.15, there may be some overplotting

![](Course_assignemtn_2_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
head(prediction1)
```

    ##  1 21 22 23 25 26 
    ##  A  A  A  A  A  A 
    ## Levels: A B C D E

``` r
head(testtrain$classe)
```

    ## [1] A A A A A A
    ## Levels: A B C D E

``` r
# Confusion matrix
matrix1 <- confusionMatrix(prediction1, testtrain$classe)
matrix1
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1235  157   16   50   20
    ##          B   55  568   73   80  102
    ##          C   44  125  690  118  116
    ##          D   41   64   50  508   38
    ##          E   20   35   26   48  625
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7394          
    ##                  95% CI : (0.7269, 0.7516)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6697          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8853   0.5985   0.8070   0.6318   0.6937
    ## Specificity            0.9307   0.9216   0.9005   0.9529   0.9678
    ## Pos Pred Value         0.8356   0.6469   0.6313   0.7247   0.8289
    ## Neg Pred Value         0.9533   0.9054   0.9567   0.9296   0.9335
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2518   0.1158   0.1407   0.1036   0.1274
    ## Detection Prevalence   0.3014   0.1790   0.2229   0.1429   0.1538
    ## Balanced Accuracy      0.9080   0.7601   0.8537   0.7924   0.8307

This model is only 75% accurate, I want to try another model

Model 2 - Random Forresting
===========================

``` r
model2 <- randomForest(classe ~., data = train, type = "class")
prediction2 <- predict(model2, testtrain, type = "class")

# Confusion Matrix
matrix2 <- confusionMatrix(prediction2, testtrain$classe)
matrix2
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    3    0    0    0
    ##          B    0  943   10    0    0
    ##          C    0    3  844    5    0
    ##          D    0    0    1  799    0
    ##          E    0    0    0    0  901
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9955          
    ##                  95% CI : (0.9932, 0.9972)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9943          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9937   0.9871   0.9938   1.0000
    ## Specificity            0.9991   0.9975   0.9980   0.9998   1.0000
    ## Pos Pred Value         0.9979   0.9895   0.9906   0.9988   1.0000
    ## Neg Pred Value         1.0000   0.9985   0.9973   0.9988   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1923   0.1721   0.1629   0.1837
    ## Detection Prevalence   0.2851   0.1943   0.1737   0.1631   0.1837
    ## Balanced Accuracy      0.9996   0.9956   0.9926   0.9968   1.0000

``` r
# 99.5% accurate
```

This is a much better model **99.5% accurate** \# Quiz prediction

``` r
quizdata <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
quizdata <-quizdata[,colSums(is.na(quizdata)) == 0]
quizdata <-quizdata[,-c(1:7)]

quizprediction <- predict(model2, quizdata, type = "class")
quizprediction
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Ending analysis and conclusion
==============================

In my model, I used cross validation by splitting the training data into 75% for training and 25% for testing. I created a model with the 75% and tested it on the remaining 25% for which I already knew the true values it was going to predict. I found that using Random forresting was the best model as it was 99.5% accurate.

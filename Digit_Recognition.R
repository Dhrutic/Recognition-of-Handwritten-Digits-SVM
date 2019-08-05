############################ SVM Digit Recogniser #################################

# Number of Attributes: 785 


#Loading Neccessary libraries

library(kernlab)
library(readr)
library(caret)
library(readr)
library(gridExtra)
library(Rtsne)


#Loading Data

DR_test <- read_csv("mnist_test.csv")
DR_train<- read_csv("mnist_train.csv")

#Understanding Dimensions

dim(DR_test)
dim(DR_train)

#Structure of the dataset

str(DR_train)
str(DR_test)

#printing first few rows

head(DR_train)
head(DR_test)


#Exploring the data

summary(DR_train)
summary(DR_test)



#checking missing value

sapply(DR_train, function(x) sum(is.na(x)))
sapply(DR_test, function(x) sum(is.na(x)))

sapply(DR_train[1, ], class)
sapply(DR_test[1, ], class)


DR_test<-as.data.frame(DR_test)
DR_train<-as.data.frame(DR_train)


#Labelling same column names for train and test dataset
DR_train[, 1] <- as.factor(DR_train[, 1])
colnames(DR_train) <- c("Y", paste("X.", 1:784, sep = ""))

class(DR_train[, 1])
#[1] "factor"

levels(DR_train[, 1])
# [1] "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"




DR_test[, 1] <- as.factor(DR_test[, 1])
colnames(DR_test) <- c("Y", paste("X.", 1:784, sep = ""))
class(DR_test[, 1])
#[1] "factor"

levels(DR_test[, 1])
# [1] "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"


#Duplicates
sum(duplicated(DR_test))
sum(duplicated(DR_train))




#scaling the data
max(DR_train[ ,2:ncol(DR_train)])
#255
train_dr <- cbind(label = DR_train[ ,1], DR_train[ , 2:ncol(DR_train)]/255)
test_dr <- cbind(label = DR_test[ ,1], DR_test[ , 2:ncol(DR_test)]/255)



# Split the data into train and test set
#Only 20 % of the data from the training set will be consider for model builing due to restrcited computation power
set.seed(1)
train.indices = sample(1:nrow(train_dr),0.2*nrow(train_dr)
train <- train_dr[train.indices,]


#Reducing dimensionality
tsne <- Rtsne(train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)

# display the results of T-sne
colors = rainbow(length(unique(train$label)))
names(colors) = unique(train$label)
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=train$label, col=colors[train$label])


# plotting the digits

colors<-c('white','purple')
 cus_col<-colorRampPalette(colors=colors)
 
 par(mfrow=c(4,3),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
 
 all_img<-array(dim=c(10,28*28))
for(d in 0:9)
  {
       print(d)
         all_img[d+1,]<-apply(train[train[,1]==d,-1],2,sum)
         all_img[d+1,]<-all_img[d+1,]/max(all_img[d+1,])*255
         
           z<-array(all_img[d+1,],dim=c(28,28))
           z<-z[,28:1] 
           image(1:28,1:28,z,main=d,col=cus_col(256))
}
      
  
#Constructing Model

#Using Linear Kern
LRM <- ksvm(label~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(LRM, test_dr)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test_dr$label)


#Linear Kernel Accuracy : 91%         


#Using RBF Kernel
RBFM <- ksvm(label~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBFM<- predict(RBFM, test_dr)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBFM,test_dr$label)


                                       
#RBF Kernel Accuracy : 96%         
# RBF lernel gives better accuracy

#Optimization using hyperparameter
trainControl <- trainControl(method="cv", number=5)
# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.01,0.025, 0.05,0.075), .C=c(0.1,0.5,1,2))


# 1000 records were considered for optimization
fit.svm <- train(label ~ ., data = head(train, 1000), method="svmRadial", metric=metric, tuneGrid=grid, trControl=trainControl)
print(fit.svm)
#sigma  C     Accuracy   Kappa  
#0.025  0.50  0.8929538  0.88080408

plot(fit.svm)


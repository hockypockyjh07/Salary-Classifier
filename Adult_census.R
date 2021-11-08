library('dplyr')
library('ggplot2')
library('janitor')
adult = read.csv('adult.csv')

# preprocessing 
#removing redundant or unknown variables
clean_adult <- adult %>% select(-fnlwgt,-education,-relationship)

#cleaning outliers using interquartile rule(IQR)
### age ###
summary(clean_adult$age)
age_Q1 = 28
age_Q3 = 48
age_IQR = age_Q3 - age_Q1
#age_IQR = 20

#find lower bound
lowerB_age = age_Q1 - (age_IQR * 1.5)
#lowerB_age = -2

#find upper bound
UpperB_age = age_Q3 + (age_IQR * 1.5)
#UpperB_age = 78

clean_adult  <- clean_adult %>% filter( age >=-2|age <= 78 )


### education.num ###
summary(clean_adult$education.num)
edunum_Q1 = 9
edunum_Q3 = 12
edunum_IQR = edunum_Q3 - edunum_Q1
#edunum_IQR = 3

#find lower bound
lowerB_edunum = edunum_Q1 - (edunum_IQR * 1.5)
#lowerB_edunum = 4.5

#find upper bound
UpperB_edunum = edunum_Q3 + (edunum_IQR * 1.5)
#UpperB_edunum = 16.5

clean_adult  <- clean_adult %>% filter(education.num >= 4.5|education.num<=16.5)


### capital.gain ###
summary(clean_adult$capital.gain)
#capitalg_plot = ggplot(adult, aes(x=capital.gain)) + geom_boxplot()
#capitalg_plot
clean_adult <- clean_adult %>% filter(capital.gain<99999)

### Reclassifying variables###
clean_adult$occupation <- gsub("?", "Unknown",clean_adult$occupation, fixed = T )
clean_adult$occupation <- factor(clean_adult$occupation)
levels(clean_adult$occupation)[c(8,9,11)] = 'Service'

clean_adult$workclass <- gsub("?", "Unknown",clean_adult$workclass, fixed = T )
clean_adult$workclass <- factor(clean_adult$workclass)
levels(clean_adult$workclass)[c(1,2,7)] = 'Government'
levels(clean_adult$workclass)[c(4,5)] = 'Self-emp'

clean_adult$marital.status <- factor(clean_adult$marital.status)
levels(clean_adult$marital.status)[c(2:4)] = 'Married'

clean_adult$native.country <- factor(clean_adult$native.country)
clean_adult %>% tabyl(native.country) %>% adorn_pct_formatting(digits=0) %>%arrange(desc(n))
# 90% of the table consists of US, so change values to US and non-US
levels(clean_adult$native.country)[c(1:39,41,42)] = 'non-United-States'

clean_adult$sex <- factor(clean_adult$sex)
clean_adult$race <- factor(clean_adult$race)
clean_adult$income <- factor(clean_adult$income)

### Normalization of numeric variables ###
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

clean_adult$age<-min_max_norm(clean_adult$age)
clean_adult$education.num<-min_max_norm(clean_adult$education.num)
clean_adult$capital.gain<-min_max_norm(clean_adult$capital.gain)
clean_adult$capital.loss<-min_max_norm(clean_adult$capital.loss)
clean_adult$hours.per.week<-min_max_norm(clean_adult$hours.per.week)

# set random seed
set.seed(14764)
# generate random value, based on the rows of cleaned dataset 
gp<- runif(nrow(clean_adult))
# use random number to suffle the dataset
clean_adult<- clean_adult[order(gp),]
### Train/Test data set ###
#dividing into 80:20, training and test data set
create_train_test <- function(data, size = 0.8, train = TRUE) {
  n_row = nrow(data)
  total_row = size * n_row
  train_sample <- 1: total_row
  if (train == TRUE) {
    return (data[train_sample, ])
  } else {
    return (data[-train_sample, ])
  }
}

data_train <- create_train_test(clean_adult, 0.8, train = TRUE)
data_test <- create_train_test(clean_adult, 0.8, train = FALSE)

dim(data_train)
dim(data_test)
prop.table(table(data_train$income))
prop.table(table(data_test$income))

# Preprocessing for KNN
# create dummy variable for categorical variables
library('caret')
dmy = dummyVars(" ~ .", data = clean_adult)
dmy_adult = data.frame(predict(dmy, newdata = clean_adult))

# need to remove the additional column for income 
dmy_adult = dmy_adult[-39] # removing  column 39  which is income <=50k
names(dmy_adult)[names(dmy_adult) == "income..50K"] = "income.more.50k" # renaming  the income>50k column

data_train_knn <- create_train_test(dmy_adult, 0.8, train = TRUE)
data_test_knn <- create_train_test(dmy_adult, 0.8, train = FALSE)




# KNN
library('class')

knn_pred_5 = knn(data_train_knn[-39], data_test_knn[-39], data_train_knn$income.more.50k, k=5)
summary(knn_pred_5)

knn_pred_10 = knn(data_train_knn[-39], data_test_knn[-39], data_train_knn$income.more.50k, k=10)
summary(knn_pred_10)

knn_pred_15 = knn(data_train_knn[-39], data_test_knn[-39], data_train_knn$income.more.50k, k=15)
summary(knn_pred_15)

# Confusion matrix for k=5
cm_5 = table(data_test_knn[, 39], knn_pred_5)
cm_5

# Confusion matrix for k=10
cm_10 = table(data_test_knn[, 39], knn_pred_10)
cm_10

# Confusion matrix for k=15
cm_15 = table(data_test_knn[, 39], knn_pred_15)
cm_15


# elbow method to choose k value
i = 1
k_model_list =1
for(i in 1:20){
  knn_model = knn(data_train_knn[-39], data_test_knn[-39], data_train_knn$income.more.50k, k=i)
  k_model_list[i] = 100* sum(data_test_knn$income.more.50k == knn_model)/nrow(data_test_knn)
}

for(i in 1: length(k_model_list)){
  cat('k', i, '=', k_model_list[i], '\n')
}

plot(k_model_list, type = "b", xlab="K-values", ylab = "Accuracy", spread.scale)


# Decision Tree
library(plyr)
library(rpart)
library(rpart.plot)

# Creating the tree model
tr_model <- rpart(income ~., data = data_train, method="class")
rpart.plot(tr_model)

# Predict with test data
tree_pred = predict(tr_model, newdata = data_test[-12], type = "class")

# Confusion matrix of prediction
cm_tree = table(Actual = data_test[,12],Predicted = tree_pred)
cm_tree


# Naive Bayes
library(e1071)
classifier_NB= naiveBayes(income ~ ., data=data_train, method="class")

#Predicting the Test set results
y_predNB = predict(classifier_NB, newdata=data_test[-12], method="class")
summary(y_predNB)

# Confusion Matrix of prediction
cm_nb = table(Actual = data_test[,12],Predicted = y_predNB)
cm_nb




















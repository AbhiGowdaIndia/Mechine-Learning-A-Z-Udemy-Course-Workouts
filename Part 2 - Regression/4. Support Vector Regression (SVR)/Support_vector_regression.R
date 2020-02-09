# S V R

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset=dataset[2:3]
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#Fittng linear regression to the dataset
#install.packages('e1071')
library(e1071)
regressor=svm(formula=Salary~.,
              data = dataset,
              type='eps-regression')

#Visualizing the linear regression results
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color="red")+
  geom_line(aes(x=dataset$Level,y=predict(regressor,newdata = dataset)))+
  ggtitle("Truth or Bluff (Linear Regression")+
  xlab("Level")+
  ylab("Salary")

#Predicting a new result with Linear regression
y_predict=predict(regressor,data.frame(Level=6.5))



# Decision tree regression

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

#Fittng Decision tree regression to the dataset
install.packages('rpart')
library(rpart)
regressor=rpart(formula = Salary~.,
           data = dataset,control = rpart.control(minsplit = 1))

#Visualizing the Decision tree regression results
x_grid=seq(min(dataset$Level),max(dataset$Level),0.01)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
           color="red")+
  geom_line(aes(x=x_grid,y=predict(regressor,newdata = data.frame(Level=x_grid))),
            color="blue")+
  ggtitle("Truth or Bluff (Decision tree Regression")+
  xlab("Level")+
  ylab("Salary")


#Predicting a new result with Decision tree regression
y_predict=predict(regressor,data.frame(Level=6.5))




library(glmnet)
library(cli)
library(tidyverse)
library(modelr)
library(broom)
library(ISLR)
library(pscl)
library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)
library(ISLR)
library(mice)
library(lmtest)
library(car)
library(caret)
library(nlme)

data <- Real.estate[c("transaction.date","house.age","metro_dist",
                      "market","latitude","longitude","price")]
names(data)
nrow(data)
View(data)

set.seed(150)
sampleIndex <- sample(1:nrow(data), size = 0.8*nrow(data))
trainset <- data[sampleIndex,]
testset <- data[-sampleIndex,]
nrow(trainset) ; nrow(testset)

cor(trainset)
# Metro_dist - price <- -0.65841050, meaning the farther the house is from the metro, the lower the price.

sum(is.na(trainset))
md.pattern(data)
# As we can see, there are no missing observations.

model1 <- lm(price~., data = trainset)
summary(model1)
# p-value: < 2.2e-16, the model is significant, and all variables except "longitude" are significant.

model2 <- lm(price~transaction.date + house.age + metro_dist +market +
               latitude, data = trainset)
summary(model2) # Adjusted R-squared has slightly increased.

# Now let's compare the significance of these two models using AIC and BIC.
AIC(model1, k=8)
AIC(model2, k=7)         # As we can see, model2 is better than model1.
BIC(model1)              # This is evident.
BIC(model2)

head(testset)
# Therefore, let's redefine our train and test sets without the "longitude" variable.
trainset2 <- trainset[,-6]
testset2 <- testset[,-6]
nrow(testset2) ; nrow(trainset2)
head(testset2)

# Predictions for price.
predictions <- predict(model2, testset2)
head(predictions)

## Outlier Detection
standardized_residuals <- rstandard(model2)
summary(standardized_residuals)
thresholdIndex <- which(abs(standardized_residuals) > 2)
length(thresholdIndex)

# Cook's Distance measures the influence of each observation on the model's predictions.
dist <- cooks.distance(model2)
threshold1 <- mean(dist)*3
threshold2 <- 4/length(dist)
threshold1 ; threshold2

threshold1Index <- which(dist > threshold1)
threshold2Index <- which(dist > threshold2)
length(threshold1Index)
length(threshold2Index)

outliers <- which(dist > threshold2 & abs(standardized_residuals) > 2)

trainsetrem <- trainset2[-outliers,]
nrow(trainset) # original
nrow(trainsetrem) # updated

# At this point, let's try to estimate the variance model of model2.
plot(model2$fitted.values,model2$residuals)
# Since we can't interpret much from this plot, let's perform some tests.
bptest(model2,data = trainset2)
# H0: No heteroskedasticity (residual variance is constant).
# H1: There is heteroskedasticity (residual variance is not constant).
# The result shows that there is no evidence to reject H0 > 0.05. Thus, there is no heteroskedasticity.

# Now let's check if the residuals follow a normal distribution.
qqnorm(residuals(model2),ylab = "residuals")
qqline(residuals(model2),col="red")
# From this plot, we cannot confirm normality.
shapiro.test(residuals(model2)) 
# As shown here, p-value < 0.05, so normality does not exist. This is unfavorable for model significance.

# Let's now check for autocorrelation in residuals.
dwtest(model2)
# H0: Residuals are independent (no autocorrelation).
# H1: Residuals are autocorrelated.
# The result is desirable for the model.

vif(model2) 
# 1 < VIF < 5: Acceptable level of multicollinearity.
# In this case, there is no strong relationship between the variables in the model, and the results can be considered reliable.


# -R2-
R2(predictions,testset2$price)
# Here, we see that R2 is 0.7797011, indicating the model explains most of the price variation.

# -RMSE-MAE- These metrics show the average deviation of model predictions from actual values.
RMSE(predictions,testset2$price) 
MAE(predictions,testset2$price)
# The average error of price predictions.

# Now let's create a new model using the updated trainset.
model3 <- lm(price~., data = trainsetrem)
summary(model3) ; summary(model2)
# Comparing these two models, model3 represents a larger portion of the dataset and has a lower residual sum of squares.

# Now let's analyze these two models using some metrics.
AIC(model2, k=6)
AIC(model3, k=6)       # As shown, model3 has lower AIC and BIC values compared to model2.
BIC(model2)            # This indicates that model3 is a better model.
BIC(model3)

vif(model3) 
# As expected, all VIF values are below 10.

# To compare R2, RMSE, and MAE values:
predictions3 <- predict(model3,testset2) 
R2(predictions3,testset2$price)    
RMSE(predictions3,testset2$price)  
MAE(predictions3,testset2$price)   

predictions2 <- predict(model2,testset)
R2(predictions2,testset2$price)    
RMSE(predictions2,testset2$price)  
MAE(predictions2,testset2$price)   
# From these results, although not significantly, we can conclude that model3 is a better model.


bptest(model3,data = trainsetrem)
# As shown here, there is evidence of heteroskedasticity.
shapiro.test(residuals(model3))
# There is insufficient evidence to reject H0. Residuals are normally distributed.
dwtest(model3)
# Residuals are independent, as they should be.
vif(model3)
# VIF values are also as expected.


# Transforming the dependent variable price using square root to reduce heteroskedasticity.

#### FINAL MODEL 
model_sqrt <- lm(sqrt(price) ~ ., data = trainsetrem)
summary(model_sqrt)

dist <- cooks.distance(model3)
threshold1 <- mean(dist)*3
threshold2 <- 4/length(dist)
threshold1 ; threshold2

threshold1Index <- which(dist > threshold1)
threshold2Index <- which(dist > threshold2)
length(threshold1Index)
length(threshold2Index)

outliers <- which(dist[1:length(standardized_residuals)] > threshold2 & 
                    abs(standardized_residuals) > 2)


plot(model_sqrt$fitted.values,model3$residuals)

predictions1 <- predict(model_sqrt, testset2)
head(predictions1)

bptest(model_sqrt,data = trainsetrem)
shapiro.test(residuals(model_sqrt))
dwtest(model_sqrt)
vif(model_sqrt)


AIC(model_sqrt, k=6) 
AIC(model3, k=6) 

BIC(modelsqrt)       
BIC(model3) 

R2(predictions1,testset$price)       
R2(predictions3,testset2$price)   


RMSE((predictions1)^2, testset$price) 
RMSE(predictions3,testset2$price)     


MAE((predictions1)^2, testset$price)  
MAE(predictions3,testset2$price)      


# This way, we have built a more meaningful model, and we can confirm this upon testing.
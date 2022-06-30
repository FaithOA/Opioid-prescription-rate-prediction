library(tidyverse)
library(dplyr)
library(ggplot2)
library(psych)
install.packages('caret')
selected <- read.csv("/Users/Faith/Tables/selectedfin.csv")


#factorizing the USBORN column to 0 - Not US-born, 1 - US-born, 2- Place of birth unknown
selected$usborn <- factor(selected$usborn)
#test-train split
split1<- sample(c(rep(0, 0.7 * nrow(selected)), rep(1, 0.3 * nrow(selected))))
table(split1)
train <- selected[split1 == 0,]   
test <-selected[split1 == 1,] 
train_labels <- train$SCORE
test_labels <- test$SCORE

#running a logistic regression model on the train data
ml = glm(SCORE ~ ageleavegme+odpc+gdr+usborn+specgrp+ume_country+region+gmegrad_group+umegrad_group+gmeregion+tocr_cy, data = train, family='binomial')
#model summary
summary(ml)
#McFadden test
install.packages('pscl')
pscl::pR2(ml)["McFadden"]
#dividing into high prescribing rate and low prescribing
selected.highvol <- subset(selected, SCORE == 1)
selected.highvol
selected.lowvol <- subset(selected, SCORE == 0)
#converting the total opioid claim rate to numeric values
x <- selected.highvol['tocr']
x_num <-as.numeric(unlist(x))
x_num
y <- selected.lowvol['tocr']
y_num <- as.numeric(unlist(y))

median(x_num)
summary(x_num)
median(y_num)
summary(y_num)
#Running the wilcoxon test to compare total opioid claim rates between the high-rate prescribers and low-rate prescribers
wilcox.test(x_num, y_num,paired=FALSE)
#variable importance for the logistic regression model
caret::varImp(ml)
#Odds ratios and confidence intervals of the LR model
confint(ml)
exp(cbind(OR = coef(ml), confint(ml)))



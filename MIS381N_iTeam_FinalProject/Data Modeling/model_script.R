
#Lab4: Multiple Linear Regression
library(SDSRegressionR)
library(tidyverse)
library(car)
#import data...
char <- read_csv("people_charity_finale.csv")

#clean data..
char <- char %>% 
  filter(Raised %not in% c("kr2,400"))
# char$raised <- log(as.numeric(char$raised))
char$goal <- log(as.numeric(char$goal))
char$people <- as.numeric(char$people)



#model

#conmputer_vision
x <- lm(Raised ~ goal + shares + days + people_1, data=char, na.rm=TRUE)
summary(x)
#topic_1
x <- lm(Raised ~ goal + shares + days + team, data=char, na.rm=TRUE)
summary(x)
#topic_2
x <- lm(Raised ~ goal + shares + days + env, data=char, na.rm=TRUE)
summary(x)
#topic_3
x <- lm(Raised ~ goal + shares + days + product, data=char, na.rm=TRUE)
summary(x)
#topic_4
x <- lm(Raised ~ goal + shares + days + face, data=char, na.rm=TRUE)
summary(x)

#import data...
bus <- read_csv("people_business_finale.csv")

#clean data..
bus <- bus %>% 
  filter(Raised %not in% c("kr2,400"))
# bus$raised <- log(bus$raised)
bus$goal <- log(as.numeric(bus$goal))
bus$people <- as.numeric(bus$people)



#model

#computer_vision
x <- lm(Raised ~ goal + shares + days + people_1, data=bus, na.rm=TRUE)
summary(x)
#topic_1
x <- lm(Raised ~ goal + shares + days + amb, data=bus, na.rm=TRUE)
summary(x)
#topic_2
x <- lm(Raised ~ goal + shares + days + human, data=bus, na.rm=TRUE)
summary(x)
#topic_3
x <- lm(Raised ~ goal + shares + days + objects, data=bus, na.rm=TRUE)
summary(x)
#topic_4
x <- lm(Raised ~ goal + shares + days + products, data=bus, na.rm=TRUE)
summary(x)


q_mod_2 <- lm(Raised ~ goal + shares + days + people_1 + amb + human + objects + products, data=bus, na.rm=TRUE)
summary(q_mod_2)








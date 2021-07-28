if(!require(tidyverse)) install.packages("tidyverse")
if(!require(lubridate)) install.packages("lubridate")
if(!require(caret)) install.packages("caret")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(rpart)) install.packages("rpart")
library(tidyverse)
library(caret)
library(lubridate)
library(gridExtra)
library(rpart)

gc<-read.csv("https://raw.githubusercontent.com/Ltaiefgit/car_price/main/autoscout24-germany-dataset.csv")
str(gc)
#########    Data Processing    #########################

###checking missing values
colSums(is.na(gc))
colSums(is.na(gc)==T|gc=="")
##Delete rows with missing value
gc<-gc%>%filter(is.na(hp)!=T&model!=""&gear!="")


missing_model<- gc%>%filter(is.na(hp)==T)%>%.$model

gc%>%group_by(offerType)%>%summarise(min=min(mileage), max=max(mileage))
###deleting the Rows with missing values
gc<- na.omit(gc)
##Checking the classification features
summary(gc$gear)
summary(gc$fuel)
summary(gc$model)
summary(gc$make)


#the age of the vehicle in years
gc<-gc%>%mutate(car_age=2021-year)%>%select(-year)
###mileage ratio depending on the average german car mileage
gc<-gc%>%mutate(mileage_ratio=mileage/14000-car_age)
####################data visualization##################################
##price~make
gc%>%ggplot(aes(price,make))+
  geom_boxplot()
# High price example
gc%>%filter(make=="Maybach")

#####remove the "far outliers"(Tukey definition): make
gc<-gc%>%group_by(make)%>%
  mutate(iqr=IQR(price),min=quantile(price, .25)-3*iqr,max=quantile(price, .75)+3*iqr)%>%
  filter(price>min&price<max)
gc%>%ggplot(aes(price,make))+
  geom_boxplot()

##price~mileage
gc%>%ggplot(aes(mileage, price))+
  geom_point()
##price~mileage ratio
gc%>%filter(mileage!=0)%>%ggplot(aes(mileage_ratio,price))+
  geom_point()

gc%>%ggplot(aes(round(mileage,-4)))+
 geom_bar()
round(gc$mileage,-2)

##gear
gc%>%ggplot(aes(gear))+
  geom_bar()
gc%>%ggplot(aes(hp,price))+
  geom_point()+
  facet_grid(.~gear)+
  geom_smooth()
##offertype
gc%>%ggplot(aes(offerType, price))+
  geom_boxplot()
gc%>%ggplot(aes(offerType))+
  geom_bar()
gc%>%ggplot(aes(hp,price))+
  geom_point()+
  facet_grid(.~offerType)+
  geom_smooth()
##price~horsepower
gc%>%ggplot(aes(hp, price))+
  geom_point()+
  geom_smooth()
#horsepower distribution
gc%>%ggplot(aes(hp))+
  geom_bar()
####age
gc%>%mutate(car_age=as.factor(car_age))%>%ggplot(aes(car_age, price))+
  geom_boxplot()
gc%>%ggplot(aes(car_age))+
  geom_bar()
####mileage
gc%>%ggplot(aes(mileage, price))+
  geom_point()
gc%>%ggplot(aes(mileage))+
  geom_density()
gc%>%ggplot(aes(mileage_ratio))+
  geom_density()
########fuels:
gc%>%ggplot(aes(hp,price))+
  geom_point()+
  facet_grid(.~fuel)+
  geom_smooth()
#####cng
gc%>%filter(fuel=="CNG")%>%ggplot(aes(hp,price))+
  geom_boxplot()
#####diesel
gc%>%filter(fuel=="Diesel"|fuel=="Electric/Diesel")%>%ggplot(aes(hp,price))+
  geom_boxplot()
#####gas
gc%>%filter(fuel=="Gasoline"|fuel=="Electric/Gasoline")%>%ggplot(aes(hp,price))+
  geom_boxplot()
#####electric
gc%>%filter(fuel=="Elecctric"|fuel=="Electric/Gasoline"|fuel=="Electric/Diesel")%>%ggplot(aes(hp,price))+
  geom_boxplot()
#####lpg
gc%>%filter(fuel=="LPG")%>%ggplot(aes(hp,price))+
  geom_boxplot()

gc%>%ggplot(aes(gear,price))+
  geom_boxplot()
###########make
gc%>%ggplot(aes(model,price))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

make<-gc%>%group_by(make)%>%summarise(average_price=mean(price), min=min(price), max=max(price),sample_size=n())
model<-gc%>%group_by(model)%>%summarise(average_price=mean(price), min=min(price), max=max(price),sample_size=n())




#########   ML algorithm      #########################
##seperate the dataset in test and train sets.
set.seed(1, sample.kind = "Rounding")
test_index<- createDataPartition(y = gc$price, times = 1, p = 0.1, list = FALSE)
train_set<- gc[-test_index,]
temp <- gc[test_index,]
test_set<- train_set%>%semi_join(temp, by="make")%>%semi_join(temp, by="model")
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(test_index, temp,removed)
###linear model
fit_lm<- train(price~., data=train_set, method="lm")#fitting the model
p_lm<-predict(fit_lm, test_set)#prediction
RMSE(p_lm,test_set$price)#rmse
##polynomial fitting
fit_poly<- loess(price~hp+mileage_ratio, data=train_set)#fitting the model
p_poly<-predict(fit_poly, test_set)#prediction
RMSE(p_poly,test_set$price)#rmse

###rpart 
train_rpart <- train(price ~ ., method = "rpart",tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                     data =train_set)#fitting the model and choosing the complexity parameter 
ggplot(train_rpart)

plot(train_rpart$finalModel)#plotting the tree
text(train_rpart, cex = 0.75)
p_rp<-predict(train_rpart,test_set)#prediction
RMSE(p_rp, test_set$price)#rmse

#####matrix factorization algorithm
#average car price
mu_hat<-mean(train_set$price)
#Price prediction according to:
#make
average_p_make<-train_set%>%group_by(make)%>%summarise(b_m=mean(price-mu_hat))
qplot(b_m, data=average_p_make)
prediction_make<- mu_hat+test_set%>%left_join(average_p_make, by="make")%>%pull(b_m)
RMSE(prediction_make,test_set$price)
#horsepower+make
average_mmhp<-train_set%>%left_join(average_p_make, by="make")%>%group_by(hp)%>%summarise(b_h=mean(price-mu_hat-b_m))
predicted_mmhp<-test_set%>%left_join(average_p_make, by="make")%>%left_join(average_mmhp, by="hp")%>%mutate(p=mu_hat+b_m+b_h)%>%pull(p)
qplot(b_h, data=average_mmhp)
RMSE(predicted_mmhp,test_set$price)
#car mileage_ratio+horsepower+make
average_mmhpmr<- train_set%>%left_join(average_p_make, by="make")%>%left_join(average_mmhp, by = "hp")%>%group_by(mileage_ratio)%>%summarise(b_r=mean(price-mu_hat-b_m-b_h))
qplot(b_r, data=average_mmhpmr)
prediction_mmhpmr<-test_set%>%left_join(average_p_make, by="make")%>%left_join(average_mmhp, by="hp")%>%left_join(average_mmhpmr, by="mileage_ratio")%>%mutate(p=mu_hat+b_m+b_h+b_r)%>%pull(p)
RMSE(prediction_mmhpmr, test_set$price)
#fuel+mileage_ratio+horsepower+make
average_mmhpmrf<- train_set%>%left_join(average_p_make, by="make")%>%left_join(average_mmhp, by = "hp")%>%left_join(average_mmhpmr, by = "mileage_ratio")%>%group_by(fuel)%>%summarise(b_f=mean(price-mu_hat-b_m-b_h-b_r))
qplot(b_f, data=average_mmhpmrf)
prediction_mmhpmrf<-test_set%>%left_join(average_p_make, by="make")%>%left_join(average_mmhp, by="hp")%>%left_join(average_mmhpmr, by="mileage_ratio")%>%left_join(average_mmhpmrf, by = "fuel")%>%mutate(p=mu_hat+b_m+b_h+b_r+b_f)%>%pull(p)
RMSE(prediction_mmhpmrf, test_set$price)



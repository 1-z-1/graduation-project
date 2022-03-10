library('rsq')
library('pROC')
setwd('C:/Users/24238/Desktop/毕业设计')
data <- xlsx::read.xlsx('./毕业论文代码/LogisticData.xlsx','Sheet1')
set.seed(513213)
index <-  sort(sample(nrow(data), nrow(data)*.75))
train <- data[index,]
test <-  data[-index,]
attach(train)
train.glm=glm(RESULT~OPEN+CLOSE+MAX+MIN+VOLUME+
                CCI+RSI+TRIX+MFI+FASTD+WR+
                MACD+SLOWD+FASTK+ROC,family=
          binomial(link='logit'),maxit=200)

# AIC Model
step(train.glm)
train.AIC=glm(formula =  RESULT ~ OPEN + CLOSE + MAX + TRIX + FASTD + FASTK, 
              family = binomial(link = "logit"), 
    maxit = 200)
summary(train.glm)
summary(train.AIC)

anova(object = train.AIC,test = "Chisq")

# Predict
test$PRE=predict(train.AIC,newdata = test,type="response")
train$PRE=predict(train.AIC,newdata = train,type="response")
roc.train=roc(test$RESULT,test$PRE)
for(i in 1:length(test$RESULT)){
  if(test[i,'PRE']>=0.6){
    test[i,'PRE'] = 1
  }
  else{
    test[i,'PRE'] = 0
  }
}
for(i in 1:length(train$RESULT)){
  if(train[i,'PRE']>=0.6){
    train[i,'PRE'] = 1
  }
  else{
    train[i,'PRE'] = 0
  }
}
rsq(train.AIC)
plot(roc.train,main="ROC of AIC Logistic Model")
accuracy_test = 1-abs(sum(test$PRE-test$RESULT)/length(test$RESULT))
accuracy_train = 1-abs(sum(train$PRE-train$RESULT)/length(train$RESULT))
detach(data)

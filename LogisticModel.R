setwd('C:/Users/24238/Desktop/毕业设计')
data <- xlsx::read.xlsx('./毕业论文代码/LogisticData.xlsx','Sheet1')
set.seed(1124)
index <-  sort(sample(nrow(data), nrow(data)*.8))
train <- data[index,]
test <-  data[-index,]
attach(train)
train.glm=glm(RESULT~OPEN+CCI+RSI+WR+MACD+SLOWD+SLOWK+FASTK,family=
              binomial(link='logit'),maxit=200)
anova(object = train.glm,test = "Chisq")
summary(train.glm)
test$PRE=predict(train.glm,newdata = test,type="response")
train$PRE=predict(train.glm,newdata = train,type="response")
roc.train=roc(test$RESULT,test$PRE)
for(i in 1:length(test$RESULT)){
  if(test[i,'PRE']>0.6){
    test[i,'PRE'] = 1
  }
  else{
    test[i,'PRE'] = 0
  }
}
for(i in 1:length(train$RESULT)){
  if(train[i,'PRE']>0.6){
    train[i,'PRE'] = 1
  }
  else{
    train[i,'PRE'] = 0
  }
}
plot(roc.train,main="model")
accuracy_test = 1-abs(sum(test$PRE-test$RESULT)/length(test$RESULT))
accuracy_train = 1-abs(sum(train$PRE-train$RESULT)/length(train$RESULT))
detach(data)
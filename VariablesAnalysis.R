setwd('C:/Users/24238/Desktop/毕业设计')
data <- xlsx::read.xlsx('./毕业论文代码/LogisticData.xlsx','Sheet1')
CCI = data$CCI
days = c(1:525)
plot(days,CCI,type='l')
lines(days,rep(100,525),col=3)
lines(days,rep(-100,525),col=3)


RSI = data$RSI
plot(days,RSI,type='l')
lines(days,rep(50,525),col='red')

MFI = data$MFI
plot(days,MFI,type = 'l')
lines(days,rep(mean(MFI),525),col='blue')
lines(days,rep(20,525),col='red')
lines(days,rep(80,525),col='red')
import json
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
import talib
import os


os.chdir('C:/Users/24238/Desktop/毕业设计/毕业论文代码')

# 请求数据
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) ',
}
with urllib.request.urlopen(
        urllib.request.Request(url='https://quotes.sina.cn/fx/api/openapi.php/BtcService.getDayKLine?symbol=btcbtcusd',
                               headers=header)) as rew:
    html = rew.read().decode('utf-8')
    priceDict = json.loads(html)
    priceDict = priceDict['result']['data'].split('|')
    date = []
    value = []
    valueStart = []
    valueMax = []
    valueMin = []
    deal = []
    for i in range(len(priceDict)):
        date.append(priceDict[i].split(',')[0])
        valueStart.append(priceDict[i].split(',')[1])
        valueMax.append(priceDict[i].split(',')[3])
        valueMin.append(priceDict[i].split(',')[2])
        value.append(priceDict[i].split(',')[4])
        deal.append(priceDict[i].split(',')[5])
data = pd.DataFrame({'date':date,'OPEN':valueStart,'MAX':valueMax,'MIN':valueMin,'CLOSE':value,'VOLUME':deal})
# 计算拓展指标
data['CCI'] = talib.CCI(data['MAX'], data['MIN'], data['CLOSE'], timeperiod=15)
data['SLOWK'],data['SLOWD'] = talib.STOCH(data['MAX'],data['MIN'],data['CLOSE'])
data['FASTK'],data['FASTD'] = talib.STOCHF(data['MAX'],data['MIN'],data['CLOSE'])
data['RSI'] = talib.RSI(data['CLOSE'], timeperiod=15)
data['WR'] = talib.WILLR(data['MAX'], data['MIN'], data['CLOSE'], timeperiod=7)
data['MACD'],signal,hist=talib.MACD(data['CLOSE'])
data["TRIX"] = talib.TRIX(data['CLOSE'],timeperiod=30)
data["ROC"] = talib.ROC(data['CLOSE'],timeperiod=10)
data["MFI"] = talib.MFI(data['MAX'], data['MIN'],data['CLOSE'],data['VOLUME'], timeperiod=14)
data['ADOSC'] = talib.ADOSC(data['MAX'],data['MIN'], data['CLOSE'], data['VOLUME'], fastperiod=3, slowperiod=10)
#准备对技术指标进行移位

# 数据存储
data.to_excel('MetaBTCData.xlsx',index=False)

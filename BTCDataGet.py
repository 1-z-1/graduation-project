import json
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
import talib
import os


os.chdir('.\Desktop\毕业设计')

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
data['CCI'] = talib.CCI(data['MAX'], data['MIN'], data['CLOSE'], timeperiod=14)
data['SLOWK'],data['SLOWD'] = talib.STOCH(data['MAX'],data['MIN'],data['CLOSE'])
data['FASTK'],data['FASTD'] = talib.STOCHF(data['MAX'],data['MIN'],data['CLOSE'])
data['RSI'] = talib.RSI(data['CLOSE'], timeperiod=7)
data['WR'] = talib.WILLR(data['MAX'], data['MIN'], data['CLOSE'], timeperiod=14)
data['MACD'],signal,hist=talib.MACD(data['CLOSE'])
# 数据存储
data.to_excel('BTCdata.xlsx')

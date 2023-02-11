#필수설치
#pip install padas
#pip install matplotlib
#pip install statsmodels

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# 데이터 불러오기
df = pd.read_csv('./data.csv')

# 현재 데이터 시각화
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
font_size = 15
plt.scatter(df['Height_'],df['Result_']) ## 원 데이터 산포도
plt.xlabel('Height_', fontsize=font_size)
plt.ylabel('Result_',fontsize=font_size)
plt.show()


#예측
#단순선형회귀모형 적합
fit = ols('Result_ ~ Height_ - 1',data=df).fit() 
print(fit.predict(exog=dict(Height_=[1.3])))



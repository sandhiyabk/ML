from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("house.csv")
dataset.head()
x=df[['Bedroom','Size','Age','Zipcode']]
y=df['Selling Price']
ct=ColumnTransformer(transformer=[('encoder',OneHotEncoder(),['Zipcode'])],remainder='passthrough')
xen=ct.fit_transform(x)
xtr,xte, ytr,yte=train_test_split(x,y,test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(xtr,ytr)
ypr=model.predict(xte)
coefficients=model.coef_
intercept=model.intercept
print("coefficients",coefficients)
print("intercept",intercept)
plt.figure(figsize=(8,6))
sns.scatterplot(x=yte,y=ypr,clor='size',s=100)
plt.plot([min(yte),max(yte)],[min(yte),max(yte)],'r--')
plt.title("Actual vs Predicted House Price")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.grid(True)
plt.tight_layout( )
plt.show()
sns.heatmap(x,corr(),annot=True,cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

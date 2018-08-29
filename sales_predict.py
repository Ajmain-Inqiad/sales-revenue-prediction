import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from matplotlib import pyplot as plt

def regression_model(model, df1, df2, name):
  x = df1[['month','year','laptop','mobile']]
  y = df1[['revenue']]
  model.fit(x,y)
  x_test = df2[['month','year','laptop','mobile']]
  y_test = df2[['revenue']]
  predictions = model.predict(x_test)
  accuracy = model.score(x_test,y_test)
  print ('Accuracy : %s' % '{0:.3%}'.format(accuracy), name)
  newarray = x_test[['month']].values
  newarray2 = newarray.ravel()
  newarray3 = x_test[['year']].values
  newarray4 = newarray3.ravel()
  newarray5 = predictions.ravel()
  newdf = pd.DataFrame({'revenue':newarray5, 'month':newarray2, 'year':newarray4})
  sns.factorplot(x="month", y="revenue", hue="year", data=newdf)
  plt.show()



data = pd.read_csv("C:\\Users\\Inqiad Ajmain\\Desktop\\testing.csv")
df = pd.DataFrame(data)
x = df[['month','year','laptop','mobile']]
y = df[['revenue']]
sns.factorplot(x="month", y="revenue", hue="year", data=df)
plt.show()
data2 = pd.read_csv("C:\\Users\\Inqiad Ajmain\\Desktop\\testingval.csv")
df2 = pd.DataFrame(data2)
clf1 = svm.SVR()
clf2 = tree.DecisionTreeRegressor()
clf3 = LinearRegression()
clf4 = GaussianNB()
clf5 = LogisticRegression()
regression_model(clf1, df, df2, "SVM")
regression_model(clf2, df, df2, "Decision Tree")
regression_model(clf3, df, df2, "Linear Regression")
regression_model(clf4, df, df2, "GaussianNB")
regression_model(clf5, df, df2, "Logistic Regression")

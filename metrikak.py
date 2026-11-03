import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
azalera = rng.uniform(30, 200, 40)
prezioa = 30000 + 400*azalera + 22*azalera**2 + rng.normal(0, 35000, 40)
df = pd.DataFrame({"azalera_m2": azalera, "prezioa": prezioa}).round()
X, y = df[["azalera_m2"]], df["prezioa"]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=42
)
print(X_train.shape, X_test.shape) # (30, 1) (10, 1)

eredua = LinearRegression().fit(X_train, y_train)
y_iragarpena = eredua.predict(X_test)
mae = mean_absolute_error(y_test, y_iragarpena)
mse = mean_squared_error(y_test, y_iragarpena)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_iragarpena)
print(mae, rmse, r2) # 71259 94097 0.900

df = pd.read_csv("etxeak.csv")

X = df[["azalera_m2"]]
y = df["prezioa"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
for gradua in [1, 2, 3, 5, 8, 12]:
    pf = PolynomialFeatures(degree=gradua, include_bias=False)
    sc = StandardScaler()
    X_tr = sc.fit_transform(pf.fit_transform(X_train))
    X_te = sc.transform(pf.transform(X_test))
    e = LinearRegression().fit(X_tr, y_train)

    print(gradua, round(e.score(X_tr, y_train), 3), # entrenamendua
        round(e.score(X_te, y_test), 3)) 


    
from sklearn.model_selection import learning_curve
tamainak, tr, va = learning_curve(LinearRegression(), X, y,
train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], cv=5)
print(tamainak) # [ 6 12 19 25 32]
print(tr.mean(axis=1)) # [0.970 0.961 0.955 0.949 0.945]
print(va.mean(axis=1)) 

#
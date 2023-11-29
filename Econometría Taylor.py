# %% [markdown]
# # Universidad Nacional Autónoma de México
# ## Facultad Economía
# ### Introducción a la Econometría
# 
# 

# %% [markdown]
# > ***Prof. Javier Galán Figueroa***

# %% [markdown]
# >> *Ejemplo de la estimación de los Mínimos Cuadrados Ordinarios Matricial con Python*

# %% [markdown]
# $ y = X{\beta} + {\upsilon} $
# 
# $ y_i = \hat{\alpha_0} + \hat{\beta}X_i + \hat{\upsilon_i} $
# 
# $ \hat{\beta} = (X'X)^{-1}X'y $

# %% [markdown]
# Para obtener el vector beta de los MCO mediante algebra lineal se hará uso de las librerias siguientes:
# * Pandas
# * Numpy

# %%
import numpy as np
import pandas as pd
import openpyxl as ox


# %%
Taylor = pd.read_excel('/Users/alfonsomarin/Desktop/Econometría/Datos/Datos Econ.xlsx')
df = pd.read_excel('/Users/alfonsomarin/Desktop/Econometría/Datos/Datos Econ.xlsx')
Taylor.head()

# %%
IGAE_GAP = df['IGAE_GAP']
TIE = df['TIE']
FIX = df['FIX']
INFLATION_GAP = df['INFLATION_GAP']
DES = df['DES']



# %% [markdown]
# Utilizar **YEAR** como index

# %%
Taylor.set_index('DATE', inplace = True)
Taylor.head()

# %% [markdown]
# Y := TIE
# X2 := Brecha del Producto (IGAE)
# X3 := Brecha de Inflación
# X4 := Tipo de Cambio FIX 
# 

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
# Taylor['IGAE_GAP'].plot.hist(color = 'brown')

# %%
boxplot1 = Taylor.plot.box(y = ['TIE'],
                            grid = True)

# # %%
boxplot1 = Taylor.plot.box(y = ['TIE', 'INFLATION_GAP'],
                           grid = True)
 
# %%
Taylor.describe()

# %% [markdown]
# $ \hat{\beta} = (X'X)^{-1}X'y $

# %%
# Generar el vector (variable) X1 al dataframe
Taylor['IGAE_GAP']=1
Taylor.head()

# %%
#Transformar el dataframe a matrices mediante "numpy"
y = Taylor ['TIE'].to_numpy()
print(y)

# %%
X = Taylor[['IGAE_GAP','INFLATION_GAP','FIX','DES']].to_numpy()
print(X)

# %% [markdown]
#  #  Para evitar la notación científica 

# %%
np.set_printoptions(suppress = True)

# %%
print(X)

# %%
#Transponer a la matrix X
trX = X.T
print (trX)

# %%
# Obtener X´X
X_X = np.dot(trX, X)
print (X_X)

# %%
# determinante de (X'X)
print(np.linalg.det(X_X))

# %%
# inversa de X_X
invX_X = np.linalg.inv(X_X)
print(invX_X)

# %%
# Obtener X'y
Xy = np.dot(trX, y)
print(Xy)

# %%
#Obtener el vector beta
beta = np.dot(invX_X, Xy)
print(beta)
print('y = 5.15604032 + 2.05290003X2 - -6.4021622X3 + 4.16735642X4 ')
# %%

# %%
# Obtener SRC = u'u = y'y - B'X'y
Try = y.T
y_y = np.dot(Try, y)
print (y_y)
# %%
# Transponer a beta
TrB = beta.T
print (TrB)

# %%
BXy = np.dot(TrB, Xy)
print (BXy)

# %%
u_u = y_y - BXy
print (u_u)

# %%
# Var(u) = u'u / n-k
Taylor.describe


# %%
n_k = 100-4
print(n_k)

# %%
var_u = u_u / n_k
print (var_u)

# %%
sd_var_u = var_u**(1/2)
print (sd_var_u)

# %%
# Obtener la matriz var_cov (beta) = var(u)**(X'x)
var_cov_beta = var_u*invX_X
print (var_cov_beta)

# %%
ee_B1 = 0.06953678**(1/2)
ee_B2 = 0.09704252**(1/2)
ee_B3 = 37.16785991**(1/2)
ee_B4 = 7.40988839**(1/2)
print (ee_B1, ee_B2, ee_B3, ee_B4 )

# %%
# Obtener el coeficiente de determinación R^2 
# R^2 = (B'X'y - n*medY^2)/(y'y - n*medY'^2)
medY2 = Taylor['TIE'].mean()**2
print (medY2)

# %%
R2 = (BXy - (20*medY2)) / (y_y - 20*medY2)
R2

# %%
#Se procede a estimar al modelo de manera directa
print(Taylor.shape)

# %%
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# %%
#Estimar el modelo 
mod1 = sm.OLS(endog = Taylor['TIE'],
             exog = Taylor [['IGAE_GAP', 'INFLATION_GAP', 'FIX','DES']])

# %%
#Obtener los resultados del modelo
est_mod1 = mod1.fit()

# %%
# presentar los resultados
print(est_mod1.summary())

# %%
#ANOVA
from scipy.stats import f_oneway

# %%
#Matriz dr corrrelación
corr_taylor = Taylor.corr()
corr_taylor

# %%
f_statistic, p_value = f_oneway(IGAE_GAP, INFLATION_GAP, FIX, DES)

# Display the resultsº
print("F-statistic:", f_statistic)
p_value_decimal = format(p_value, ".30f")
print("P-value:", p_value_decimal)

# %%
plt.scatter(IGAE_GAP, TIE, label='Relación entre TIE y brecha de Producto')

# # %%
plt.scatter(INFLATION_GAP, TIE, label='Relación entre TIE y brecha de Inflación')

# # %%
plt.scatter(FIX, TIE, label='Relación entre TIE y Tipo de cambio')

# # %%
plt.scatter(DES, TIE, label='Relación entre TIE y Desempleo')


# # %%  
# %% [markdown]
# # Universidad Nacional Autónoma de México
# ## Facultad Economía
# ### Introducción a la Econometría
#  #### Proyecto Final Entrega 1

# > *Regla de Taylor; Una aproximación empírica: El caso de la economía mexicana (2002-2023)*
# >> ***Docente Javier Galán Figueroa***
# >>> ***Camacho Elizarrarás Marín Ruíz***
import numpy as np
import pandas as pd
import openpyxl as ox
import statsmodels.api as sm

#Abrir Base de datos en excel
RT = pd.read_excel('/Users/alfonsomarin/Desktop/Everything/UNAM/5to Semestre/Intro. Econometría/T FINAL /ReglaT.xlsx')
df = pd.read_excel('/Users/alfonsomarin/Desktop/Everything/UNAM/5to Semestre/Intro. Econometría/T FINAL /ReglaT.xlsx')
RT.set_index('FECHA', inplace = True)
# Y := Tasa de Interés
# X2 := Tipo de Cambio México-Estados Unidos
# X3 :=  Brecha de Inflación
# X4 :=  Tasa de Interés USA
# X5 := Brecha de Producto
# Generar el vector (variable) X1 al dataframe
df['Intercepto']=1
#Transformar el dataframe a matrices mediante "numpy"
y = df['TI'].to_numpy()
X = df[['Intercepto','EXMXUS','FEDFUNDS','INFLATION','GDP']].to_numpy()
#  Remover notación científica 
np.set_printoptions(suppress = True)
#Transponer la matrix X
trX = X.T
# Obtener X´X
X_X = np.dot(trX, X)
# determinante de (X'X)
print("Determinante:", np.linalg.det(X_X))
# inversa de X_X
invX_X = np.linalg.inv(X_X)
# Obtener X'y
Xy = np.dot(trX, y)
#Obtener el vector beta
beta = np.dot(invX_X, Xy)
print(beta)
print('y = 4.09239085 + 0.01727235X2 -0.88053974X3 0.01778182X4 -0.02277944X5')
# Obtener SRC = u'u = y'y - B'X'y
Try = y.T
y_y = np.dot(Try, y)
# Transponer a beta
TrB = beta.T
BXy = np.dot(TrB, Xy)
u_u = y_y - BXy
# Var(u) = u'u / n-k
#Observaciones-Explicativas
n_k = 81-5
#Var
var_u = u_u / n_k
print ("Varianza:", var_u)
sd_var_u = var_u**(1/2)
print ("Raíz de Varianza:", sd_var_u)

# Obtener la matriz var_cov (beta) = var(u)**(X'x)
var_cov_beta = var_u*invX_X
print ("Matriz Error:", var_cov_beta)

# Errores
ee_B1 = -0.02028637**(1/2)
ee_B2 = 0.00095845**(1/2)
ee_B3 = -0.00003877**(1/2)
ee_B4 = -0.00000438**(1/2)
ee_B5 = 0.0011924**(1/2)
print ("Errores", ee_B1, ee_B2, ee_B3, ee_B4, ee_B5 )
#REVISAR
# Obtener el coeficiente de determinación R^2 
# R^2 = (B'X'y - n*medY^2)/(y'y - n*medY'^2)
medY2 = RT['TI'].mean()**2
R2 = (BXy - (81*medY2)) / (y_y - 81*medY2)
print ("Coeficiente de determinación:", R2)
#Estimación directa del modelo
#Estimar el modelo 
mod1 = sm.OLS(endog = df['TI'],
             exog = df[['Intercepto', 'EXMXUS', 'FEDFUNDS', 'INFLATION','GDP']])
#Obtener los resultados del modelo
est_mod1 = mod1.fit()
# Resultados
print(est_mod1.summary())
#Matriz de corrrelación
corr_BP = RT.corr()
print (corr_BP)
#Fin
# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read in our data
heart = pd.read_csv("C:/Users/LENOVO/Desktop/ML/tarea4/heart.csv")

#print(heart.head())
b=heart.describe()


#plt.figure()
##grafico de dispersion
#sns.pairplot(heart[[		"trestbps",	"thalach"	,	"oldpeak"	,"thal",	"target"]])
#plt.suptitle('Pairplot for Selected Columns')
#plt.show()



def calcular_modelo(m,b,x):
    '''Retorna el valor m*x+b correspondiente al modelo lineal'''
    return m*x+b

def gradiente_descendente(prev_theta1, prev_theta0, alpha, x, y):
    N = x.shape[0]      # Conocer la cantidad de datos

    # Gradientes: derivadas de la función de error con respecto
    # a los parámetros "theta1" y "theta0"
    theta0_actual = prev_theta0 - alpha*(1/N)*np.sum((prev_theta1*x+prev_theta0)-y)
    theta1_actual = prev_theta1 - alpha*(1/N)*np.sum(((prev_theta1*x+prev_theta0)-y)*x)
    # Actualizar los pesos usando la fórmula del gradiente descendente
    return theta1_actual, theta0_actual


def funcion_costo(y,y_):
    '''Calcula el error cuadrático medio entre el dato original (y)
       y el dato generado por el modelo (y_)'''
    N = y.shape[0]
    fcost = (1/(2*N))*np.sum((y-y_)**2)
    return fcost

#inicializamos los parametros
np.random.seed(2)
m = np.random.randn(1)[0]
b = np.random.randn(1)[0]
alpha = .00055 #Tasa de aprendizaje
i = 10000 #numero de iteraciones
fcost = np.zeros((i,1))
y=heart["thalach"]
x=heart["oldpeak"]


for i in range(i):
    # Actualizar valor de los pesos usando el gradiente descendente
    [m, b] = gradiente_descendente(m,b,alpha,x,y)

    # Calcular el valor de la predicción
    y_ = calcular_modelo(m,b,x)
   
    fcost[i] = funcion_costo(y,y_)

    print("    epsilon: {}".format(fcost[i]),"m: ",m,"b: ",b)
    #buscar el valor de epsilon mas bajo
    if fcost[i] < 60:
    	print("    epsilon: {}".format(fcost[i]),"m: ",m,"b: ",b)
    	break







m2=-6.84
b2=156
y_regr2 = m2*x+b2
m3=33.34
b3=67
y_regr3 = m3*x+b3
m4=-5.4
b4=153
y_regr4 = m4*x+b4
m5=-1.31
b5=145.7
y_regr5 = m*x+b5

plt.plot(x,y_regr2,'m',label="alpha=0.0007, 0.01, 0.085")
plt.plot(x,y_regr3,'g',label="alpha=0.000085")
plt.plot(x,y_regr4,'b',label="alpha=0.00085")


y_regr = m*x+b
plt.scatter(x,y,label="datos")
plt.plot(x,y_regr,'r',label="alpha= 0.00055")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.figure()
plt.plot(range(i),fcost)
plt.legend()
plt.show()

# regresión-lineal
 la Regresión Lineal, uno de los algoritmos básicos del Machine Learning. Veremos en detalle los conceptos de  funcion de costo y como se usa el método del Gradiente Descendente para calcular de forma automática los coeficientes de este modelo lineal.

1. - Se utilizarán los datos de análisis de los datos de predicción de infartos. 
Dataset: https://data.mendeley.com/datasets/wmhctcrt5v/1 

2. - Descargar el dataset y el codigo regresion.py

3. - Revisar la ruta de lectura de datos en el código. Cambiar ruta en caso de ser necesario

heart = pd.read_csv("C:/Users/LENOVO/Desktop/ML/tarea4/heart.csv")


4. - ejecutar programa

#Conclusiones
La regresión lineal en el aprendizaje automático es un desafío de optimización cuyo objetivo es determinar los valores de m y b para que una recta se ajuste lo más adecuadamente posible a los datos (x, y). Es importante tener en cuenta lo siguiente: la función de pérdida evalúa qué tan bien la recta se adapta a los datos proporcionados. En el contexto de la regresión lineal, esta pérdida generalmente se asocia con la función de coste. El ajuste óptimo se encuentra al determinar los valores de m y b que minimizan esta función de coste. Este ajuste ideal se logra automáticamente utilizando el algoritmo de descenso por gradiente. 

Se inicio con un numero de iteraciones muy alto 60000, sin embargo, eso produce un retraso al momento de calcular en las iteraciones, 10000 iteraciones son suficientes y permiten una convergencia “ideal”, en cuanto a la función de coste el valor de épsilon “mejor” obtenido fue de 232, se probaron diferentes valores para la tasa de aprendizaje donde la del “mejor” ajuste observado en las gráficas fue de 0.00055. 

 

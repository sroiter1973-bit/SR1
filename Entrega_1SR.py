import pandas as pd 

df_Datos=pd.read_csv(r'E:\GT2026\PythonExl\Pandas1\job_salary_prediction_dataset.csv')
"""
# Entrega 1 BootcampDataScience_Samuel Roiter Vélez
El conjunto de datos elegido para la entrega, contiene 250,000 registros de empleos, diseñado para predicción de salarios, análisis de datos y proyectos de aprendizaje automático.

Incluye información sobre títulos de trabajo, nivel de experiencia, formación educativa, habilidades, tipo de industria, tamaño de la empresa y ubicación laboral.
El conjunto de datos ayuda a explorar cómo diferentes factores influyen en el salario.

Puede ser útil para:
Modelos de predicción salarial
Análisis exploratorio de datos (EDA)
Prácticas de machine learning
Proyectos de visualización de datos
Información sobre salarios por formación e industria

Requisitos del taller:
• Cargar y explorar un dataset utilizando Pandas.
Se importa la base de datos = job_salary_prediction_dataset.csv
• Realizar procesos de limpieza y transformación de datos. 
    1. se aplican los filtros para detectar duplicados;
    2. se aplican filtros para detectar datos nulos;
    3. se convierten los Paises a mayusculas.
• Aplicar filtros, agrupaciones y operaciones básicas de análisis.
    1. se filtran los paises de primer mundo en comparación con en paises en desarrollo junto con los remotos;
    2. se filtran los trabajos en modalidad de presencial vs los remotos o hybridos;
    3. se promedian los años de experiencia; 
    4. se filtran por nivel educativo.
• Generar conclusiones a partir de los resultados obtenidos.
    1. se pueden observar en que estados hay mayor contratación;
    2. el desarrollo de la actividad en relación al desarrollo social; 
    3. el promedio de experiencia más apetecido; 
    4. el nivel educativo más apetedcido. 

Requisitos del taller:
• Cargar y explorar un dataset utilizando Pandas.
• Realizar procesos de limpieza y transformación de datos.
• Aplicar filtros, agrupaciones y operaciones básicas de análisis.
• Generar conclusiones a partir de los resultados obtenidos.
"""

df_Datos.info()#No de filas y columnas RangeIndex: 250000 entries, 0 to 249999
df_Datos.isna().sum()#Suma el No de datos de cada columna 
df_Datos.duplicated().sum()#Numero de duplicados non-null
#la base de datos se encuentra limpia
print(df_Datos.shape)
print (df_Datos.head())

# Filtro una sola columna de información - Esto genera una serie de pandas

print(df_Datos['location'].head())
#cambio ciudades todo a Mayusculas
df_Datos['location']=df_Datos['location'].str.upper()
print(df_Datos['location'].head())

import matplotlib.pyplot as plt
# Count occurrences of each location
counts = df_Datos['location'].value_counts()

# Plot location
plt.figure(figsize=(10, 6))
plt.bar(counts.index.astype(str), counts.values)
plt.xticks(rotation=90)
plt.xlabel("Location")
plt.ylabel("Frecuencia")
plt.title("Frecuencia de cada estado en la columna 'location'")

plt.tight_layout()
plt.show()

#Promedio de años de experiencia preferidos
df_Datos['experience_years'].mean()
print(df_Datos['experience_years'].mean())


# Nivel educativo preferido para el trabajo 

# Categorías únicas
print("Categorías únicas:")
print(df_Datos['education_level'].unique())

# Número de categorías
print("\nCantidad de categorías:")
print(df_Datos['education_level'].nunique())

# Frecuencia por categoría
print("\nFrecuencia de cada categoría:")
print(df_Datos['education_level'].value_counts())

import matplotlib.pyplot as plt

counts = df_Datos['education_level'].value_counts()

plt.figure(figsize=(8, 5))
plt.plot(counts.index.astype(str), counts.values, marker='o')
plt.xlabel("Categorías de Education Level")
plt.ylabel("Cantidad")
plt.title("Cantidad por Categoría en Education Level")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Categorías únicas
print("Categorías únicas:")
print(df_Datos['remote_work'].unique())

# Número de categorías
print("\nCantidad de categorías:")
print(df_Datos['remote_work'].nunique())

# Frecuencia por categoría
print("\nFrecuencia de cada categoría:")
print(df_Datos['remote_work'].value_counts())

import matplotlib.pyplot as plt

counts = df_Datos['remote_work'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("Categorías de remote_work")
plt.ylabel("Cantidad")
plt.title("Cantidad por Categoría en remote_work")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

'''
Conclusiones:
1. El analisis de datos es una labor apetecida tanto en paises desarrollados, como en paise en via de desarrollo; 
2. Las industrias buscan un promedio de experiencia en el campo aproximado a los 10 años; 
3. El nivel educativo preferido para la labor es de profecionales con capacitación, más no se buscan de muy alto nivel. 
4. las organizaciones no tienen preferencia en cuanto a la modalidad del trabajo remoto o presencial, en esta labor, se puede trabajar desde cualquier lugar. 
 
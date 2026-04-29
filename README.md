# Practica 1 - Modelización en Ingeniería de Datos 

## Curso 2025-26 <img src="https://images.griddo.cunef.edu/logo-cunef-universidad-1272515f-17b3-4169-8bf6-ef63bfffe920" width="190" valign="middle"> 

### Descripción del Proyecto
Este proyecto consiste en la construcción de un pipeline end-to-end de Machine Learning diseñado para predecir el riesgo de impago en préstamos bancarios. El objetivo principal es determinar, basándose en datos históricos, si un cliente será capaz de devolver su préstamo o no.

### Estructura del Proyecto

El desarrollo se divide en módulos siguiendo el estándar de ingeniería de software para Ciencia de Datos:

1. **Clase de preprocesamiento:** src/preprocessing/practica1_preprocessing.py

2. **Clase de filtrado:** src/filtering/practica1_filtering.py

3. **Notebook de referencia:** practica1_notebook.ipynb


### Benchmarking y Comparativa
Este pipeline toma como punto de partida el material didáctico proporcionado en clase por el profesor Miguel Martín. Se ha realizado un análisis crítico de la arquitectura base para implementar mejoras sustanciales en la robustez, el tratamiento de datos no lineales y la ingeniería de variables.

- Repositorio de referencia: https://github.com/mmartinb75/modelizacion_datos_2026

- Análisis comparativo detallado: He incluido un archivo llamado Miguel_VS_Lola.txt donde detallo las diferencias técnicas entre la implementación base realizada en clase y mi propuesta.


El pipeline de referencia utiliza el fichero data/variables_withExperts.xlsx 

### BIBLIOGRAFÍA Y RECURSOS:

#### Preprocessing:
- IterativeImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
- SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- OrdinalEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
- TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- RobustScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
- TextEncoder: https://skrub-data.org/stable/reference/generated/skrub.TextEncoder.html

#### Filtering:
- DropConstantFeatures: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/DropConstantFeatures.html
- DropCorrelatedFeatures: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/DropCorrelatedFeatures.html
- SelectByShuffling: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/SelectByShuffling.html

#### Modelos:
- RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- HistGradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- Maquinas de Soporte Vectorial (SVM): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

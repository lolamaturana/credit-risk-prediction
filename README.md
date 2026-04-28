# Practica 1 - Modelización en Ingeniería de Datos 

## Curso 2025-26 <img src="https://images.griddo.cunef.edu/logo-cunef-universidad-1272515f-17b3-4169-8bf6-ef63bfffe920" width="190" valign="middle"> 

He construido un pipeline end-to-end de Machine Learning para predecir si un cliente devolverá su prestamo o no (detección de impago). Este pipeline incluye preprocesamiento, filtrado y modelado, además de la comparacion del rendimiento de tres familias de modelos diferentes.

- 1. Clase de preprocesamiento:
src/preprocessing/practica1_preprocessing.py

- 2. Clase de filtrado:
src/filtering/practica1_filtering.py

- 3. Notebook de referencia: 
practica1_notebook.ipynb


El pipeline de referencia utiliza el fichero data/variables_withExperts.xlsx 

BIBLIOGRAFÍA:

Preprocessing:
- IterativeImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
- SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- OrdinalEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
- TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- RobustScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
- TextEncoder: https://skrub-data.org/stable/reference/generated/skrub.TextEncoder.html

Filtering:
- DropConstantFeatures: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/DropConstantFeatures.html
- DropCorrelatedFeatures: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/DropCorrelatedFeatures.html
- SelectByShuffling: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/SelectByShuffling.html

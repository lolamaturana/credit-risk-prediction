import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from skrub import TextEncoder


class Practica1Preprocess:

    def __init__(self, var_to_process, target): # Defina las herramientas que voy a usar y cómo están configuradas

        # Selección inicial de variables — Eliminar variables irrelevantes o que causan data leakage
        self.raw_predictors_vars = pd.read_excel(var_to_process)
        self.raw_predictors_vars = ( self.raw_predictors_vars
                                    .query("posible_predictora == 'si'")
                                    .variable
                                    .tolist())
        self.target_var = target

        # Según el estándar oficial de Scikit-Learn: Todos los hiperparámetros y la instanciación de las herramientas deben hacerse en el __init__
        
        # Herramienta para Nulos - Variables Numéricas 
        self.num_imputer = IterativeImputer(max_iter=10, random_state=42, add_indicator=True) 
        
        # Herramienta para Nulos - Variables Categóricas
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='DESCONOCIDO')

        # Herramientas para ENCODING - Variables Categóricas
        # Ordinales
        self.ordinal_cols = ['grade', 'sub_grade']
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', 
            unknown_value=-1 # Los valores imputados como DESCONOCIDO se manejan de forma segura asignándoles un valor fuera de la escala (-1).
        )
        # Nominales
        self.target_encoder = TargetEncoder(random_state=42)

        # Herramienta para Transformación de Variables Numéricas
        self.robust_scaler = RobustScaler() # Por defecto: with_centering=True / with_scaling=True / quantile_range=(25.0, 75.0)
        
        # Herramientas de Procesamiento - Texto Libre 
        self.text_enc_title = TextEncoder(model_name='intfloat/e5-small-v2', n_components=20)
        self.text_enc_desc = TextEncoder(model_name='intfloat/e5-small-v2', n_components=20)


    def fit(self,data):
        # Leer el dataframe
        df = pd.read_csv(data)

        # Separar X e Y
        self.train_X_data = df[self.raw_predictors_vars]  # x - features o variables independientes
        self.train_y_data = df[self.target_var]  # y - la variable que se a predecir
        
        ####################################
        # Eliminar columnas con muchos nulos
        ####################################

        # Calcular el porcentaje de nulos por cada variable
        size = self.train_X_data.shape[0]
        self.nulls_vars = ( (self.train_X_data.isnull().sum()/size)
                      .sort_values(ascending=False)
                      .to_frame(name="nulls_perc")
                      .reset_index() )
        
        # Eliminar las variables con >99% nulos
        self.var_with_most_nulls = ( self.nulls_vars
                               .query("nulls_perc > 0.99")["index"]
                               .tolist() )

        X_valid = self.train_X_data.drop(columns=self.var_with_most_nulls)


        #########################
        # Generación de Variables
        #########################
                
        if 'fico_range_low' in X_valid.columns and 'fico_range_high' in X_valid.columns:
            X_valid['fico_medio'] = ((X_valid['fico_range_low'] + X_valid['fico_range_high']) / 2).astype(float)
            X_valid = X_valid.drop(columns=['fico_range_low', 'fico_range_high'])
            
        if 'installment' in X_valid.columns and 'annual_inc' in X_valid.columns:
            ingreso_mensual = X_valid['annual_inc'] / 12
            X_valid['cuota_ingreso_ratio'] = np.where(ingreso_mensual > 0, 
                                                      X_valid['installment'] / ingreso_mensual, 
                                                      np.nan).astype(float)
            
        if 'revol_bal' in X_valid.columns and 'total_rev_hi_lim' in X_valid.columns:
            X_valid['balance_limite_ratio'] = np.where(X_valid['total_rev_hi_lim'] > 0, 
                                                       X_valid['revol_bal'] / X_valid['total_rev_hi_lim'], 
                                                       np.nan).astype(float)
        
        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################

        if 'earliest_cr_line' in X_valid.columns:   # comprobar que no ha sido eliminada
            X_valid['earliest_cr_line'] = pd.to_datetime(X_valid['earliest_cr_line'])
            X_valid['earliest_cr_line_year'] = X_valid['earliest_cr_line'].dt.year.astype(float)
            X_valid['earliest_cr_line_month'] = X_valid['earliest_cr_line'].dt.month.astype(float)


        #####################
        # Tipos de Variables
        #####################

        # Identificar qué variables son numéricas y cuáles categóricas (incluyendo las recién generadas)
        self.num_vars = X_valid.select_dtypes(include='number').columns.tolist()
        self.cat_vars = X_valid.select_dtypes(include='object').columns.tolist()


        ########################
        # Imputacion de missing
        ########################
        
        if self.num_vars: # si la lista tiene elementos devuelve True
            # Entrenamos el imputador múltiple (MICE) para aprender a predecir nulos - variables numéricas
            self.num_imputer.fit(X_valid[self.num_vars])

        if self.cat_vars:  # si la lista tiene elementos devuelve True
            # Imputador categórico - "DESCONOCIDO"
            self.cat_imputer.fit(X_valid[self.cat_vars]) 
            # Imputar categóricas antes de los encoders para que no vean NaNs
            X_valid[self.cat_vars] = self.cat_imputer.transform(X_valid[self.cat_vars])       
         
        
        #################################################
        # Tratamiento de Variables Categóricas (Encoding)
        #################################################

        # Proteger las fechas originales para que no entren al TargetEncoder
        temporal_cols = ['earliest_cr_line', 'issue_d']

        # Filtrar: qué va al TargetEncoder (todo lo categórico MENOS ordinales y fechas)
        self.cols_to_target_encode = [col for col in self.cat_vars 
                                      if col not in self.ordinal_cols and col not in temporal_cols]

        # Ajustar el Ordinal Encoder - variable ordinales -- grade/sub-grade
        ordinales_vivas = [col for col in self.ordinal_cols if col in self.cat_vars]
        if ordinales_vivas:
            self.ordinal_encoder.fit(X_valid[ordinales_vivas])

        # Ajustar el Target Encoder - variables de alta cardinalidad
        if self.cols_to_target_encode:
            # Variable local para binarizar solo donde hace falta
            y_binaria = self.train_y_data != 'Fully Paid'
            self.target_encoder.fit(X_valid[self.cols_to_target_encode], y_binaria)


        #######################################
        # Transformación de Variables Numéricas
        #######################################

        if self.num_vars:
            # Entrenamos el scaler sobre datos ya imputados por MICE,
            imputed_array_fit = self.num_imputer.transform(X_valid[self.num_vars])
            imputed_cols_fit = self.num_imputer.get_feature_names_out(self.num_vars)
            # Guardamos las columnas exactas a escalar (sin los indicadores de nulos)
            self.cols_to_scale = [c for c in imputed_cols_fit if not c.startswith('missingindicator_')]
            X_num_fit = pd.DataFrame(imputed_array_fit, columns=imputed_cols_fit, index=X_valid.index)
            self.robust_scaler.fit(X_num_fit[self.cols_to_scale])


        ##############################
        # Procesamiento de Texto Libre
        ##############################
        if "emp_title" in X_valid.columns:
            self.text_enc_title.fit(X_valid["emp_title"].fillna("DESCONOCIDO"))

        if "desc" in X_valid.columns:
            X_valid['desc_formated'] = np.where(
                X_valid['desc'] == 'DESCONOCIDO',
                'DESCONOCIDO',
                X_valid['desc'].astype(str).str.split('> ').str[1].str.split('<br>').str[0]
            )
            self.text_enc_desc.fit(X_valid['desc_formated'].fillna("DESCONOCIDO"))

        return self    


        
    def transform(self, data):
        # Leer el dataframe
        df = pd.read_csv(data)

        # Separar X e Y
        X_data = df[self.raw_predictors_vars] # x - features o variables independientes
        y_data = df[self.target_var] # y - la variable que se a predecir

        ####################################
        # Eliminar columnas con muchos nulos
        ####################################
        
        # Replicar la eliminación que aprendimos en el fit
        X_data = X_data.drop(columns=self.var_with_most_nulls)

        #########################
        # Generación de Variables
        #########################
        
        if 'fico_range_low' in X_data.columns and 'fico_range_high' in X_data.columns:
            X_data['fico_medio'] = ((X_data['fico_range_low'] + X_data['fico_range_high']) / 2).astype(float)
            X_data = X_data.drop(columns=['fico_range_low', 'fico_range_high'])
            
        if 'installment' in X_data.columns and 'annual_inc' in X_data.columns:
            ingreso_mensual = X_data['annual_inc'] / 12
            X_data['cuota_ingreso_ratio'] = np.where(ingreso_mensual > 0, 
                                                     X_data['installment'] / ingreso_mensual, 
                                                     np.nan).astype(float)
            
        if 'revol_bal' in X_data.columns and 'total_rev_hi_lim' in X_data.columns:
            X_data['balance_limite_ratio'] = np.where(X_data['total_rev_hi_lim'] > 0, 
                                                      X_data['revol_bal'] / X_data['total_rev_hi_lim'], 
                                                      np.nan).astype(float)
            

        ###########################################
        # Extraer mes y año de variables temporales
        ###########################################

        if 'earliest_cr_line' in X_data.columns:   
            X_data['earliest_cr_line'] = pd.to_datetime(X_data['earliest_cr_line'])
            X_data['earliest_cr_line_year'] = X_data['earliest_cr_line'].dt.year.astype(float)
            X_data['earliest_cr_line_month'] = X_data['earliest_cr_line'].dt.month.astype(float)


        #####################
        # Tipos de Variables
        #####################

        num_vars_to_impute = [col for col in self.num_vars if col in X_data.columns]
        cat_vars_to_impute = [col for col in self.cat_vars if col in X_data.columns]


        ########################
        # Imputacion de missing
        ########################

        if cat_vars_to_impute:
            X_data[cat_vars_to_impute] = self.cat_imputer.transform(X_data[cat_vars_to_impute])

        if num_vars_to_impute:
            imputed_array = self.num_imputer.transform(X_data[num_vars_to_impute])
            imputed_cols = self.num_imputer.get_feature_names_out(num_vars_to_impute)
            
            X_num_imputed = pd.DataFrame(imputed_array, columns=imputed_cols, index=X_data.index)
            
            X_data = X_data.drop(columns=num_vars_to_impute)
            X_data = pd.concat([X_data, X_num_imputed], axis=1)


        #################################################
        # Tratamiento de Variables Categóricas (Encoding)
        #################################################

        temporal_cols = ['earliest_cr_line', 'issue_d']
        
        cols_to_target_encode = [col for col in cat_vars_to_impute if col not in self.ordinal_cols and col not in temporal_cols]
        ordinales_vivas = [col for col in self.ordinal_cols if col in cat_vars_to_impute]

        if ordinales_vivas:
            X_data[ordinales_vivas] = self.ordinal_encoder.transform(X_data[ordinales_vivas])

        if cols_to_target_encode:
            X_data[cols_to_target_encode] = self.target_encoder.transform(X_data[cols_to_target_encode])

        
        #######################################
        # Transformación de Variables Numéricas 
        #######################################

        if self.num_vars:
            cols_present = [c for c in self.cols_to_scale if c in X_data.columns]
            X_data[cols_present] = self.robust_scaler.transform(X_data[cols_present])
        

        ##############################
        # Procesamiento de Texto Libre
        ##############################
        
        dfs_de_texto = []

        if "emp_title" in X_data.columns:
            texto_transformado = self.text_enc_title.transform(X_data["emp_title"].fillna("DESCONOCIDO"))
            # Le asignamos nombres de columna únicos para no colapsar el modelo luego
            nombres_cols = [f"emp_title_emb_{i}" for i in range(texto_transformado.shape[1])]
            dfs_de_texto.append(pd.DataFrame(texto_transformado, columns=nombres_cols, index=X_data.index))
            X_data = X_data.drop(columns=["emp_title"])

        if "desc" in X_data.columns:
            desc_formated = np.where(X_data['desc'] == 'DESCONOCIDO', 'DESCONOCIDO', X_data['desc'].astype(str).str.split('> ').str[1].str.split('<br>').str[0])
            texto_transformado_desc = self.text_enc_desc.transform(pd.Series(desc_formated).fillna("DESCONOCIDO"))
            nombres_cols_desc = [f"desc_emb_{i}" for i in range(texto_transformado_desc.shape[1])]
            dfs_de_texto.append(pd.DataFrame(texto_transformado_desc, columns=nombres_cols_desc, index=X_data.index))
            X_data = X_data.drop(columns=["desc"])

        if dfs_de_texto:
            X_data = pd.concat([X_data] + dfs_de_texto, axis=1)

        
        ################
        # Limpieza Final
        ################

        cols_to_drop = [c for c in ['earliest_cr_line', 'issue_d'] if c in X_data.columns]
        X_data = X_data.drop(columns=cols_to_drop)
        X_data_output = X_data.select_dtypes(exclude=['datetime64[ns]'])


        ################################
        # Transformar Variable Objetivo
        ################################
        
        y_data_out = y_data != 'Fully Paid'
        
        return X_data_output, y_data_out

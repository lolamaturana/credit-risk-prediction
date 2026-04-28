import pandas as pd
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures, SelectByShuffling
from sklearn.ensemble import RandomForestClassifier


class Practica1Filtering:
    """
    Clase que encapsula el pipeline de selección de features en 3 etapas:
      1. DropConstantFeatures: Elimina features cuasi-constantes.
            Uso tol=0.98 (más estricto que el 0.9 de clase) para no eliminar
            las columnas indicadoras de nulos (missingindicator_*) generadas por MICE,
            que aunque son poco frecuentes, aportan información sobre el patrón de missings.

      2. DropCorrelatedFeatures: Elimina correlaciones no lineales usando correlación de Spearman.
            Ya que Spearman detecta también relaciones monótonas no lineales,
            más habituales en variables financieras.

      3. SelectByShuffling: Elimina features evaluando la caída real del rendimiento al permutarlas.
            En lugar de comparar contra variables de ruido sintético, permuta cada variable
            y mide la caída real de ROC-AUC. Si al barajar una variable el rendimiento
            no cae, esa variable no aporta información útil y se elimina.

      
    Sigue el patrón fit/transform para ajustar en train y aplicar en test sin data leakage.
    """

    def __init__(self,
                 constant_tol=0.98,
                 correlation_threshold=0.80,
                 correlation_method='spearman', 
                 shuffling_scoring='roc_auc',
                 shuffling_cv=3,
                 rf_n_estimators=50,
                 rf_max_depth=5,     
                 random_state=42):

        # Paso 1: Eliminar features cuasi-constantes
        self.drop_constant = DropConstantFeatures(tol=constant_tol)

        # Paso 2: Eliminar features correlacionadas
        self.drop_correlated = DropCorrelatedFeatures(
            variables=None,
            method=correlation_method,
            threshold=correlation_threshold
        )

        # Paso 3: SelectByShuffling - baraja cada variable; si el AUC no cae significativamente, la feature es eliminada
        self.select_shuffling = SelectByShuffling(
            estimator=RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'  # dataset desbalanceado ~80/20
            ),
            scoring=shuffling_scoring,
            cv=shuffling_cv,
            random_state=random_state
        )

    def fit(self, X_data, y_data):
        """
        Ajusta los 3 filtros secuencialmente sobre los datos de entrenamiento.
        Cada filtro aprende qué features eliminar y guarda esa información
        para aplicarla luego en transform().
        """

        # Paso 1 - fit + transform para que el paso 2 reciba datos ya filtrados
        self.drop_constant.fit(X_data)
        X_no_constant = self.drop_constant.transform(X_data)
        self.n_dropped_constant = X_data.shape[1] - X_no_constant.shape[1]

        # Paso 2: fit + transform para que el paso 3 reciba datos ya filtrados
        self.drop_correlated.fit(X_no_constant)
        X_no_correlated = self.drop_correlated.transform(X_no_constant)
        self.n_dropped_correlated = X_no_constant.shape[1] - X_no_correlated.shape[1]

        # Paso 3: fit
        self.select_shuffling.fit(X_no_correlated, y_data)

        # Guardamos info para el resumen
        X_final = self.select_shuffling.transform(X_no_correlated)
        self.n_dropped_shuffling = X_no_correlated.shape[1] - X_final.shape[1]
        self.n_features_initial = X_data.shape[1]
        self.n_features_final = X_final.shape[1]
        self.selected_features = X_final.columns.tolist()

        return self

    def transform(self, X_data):
        """
        Aplica los 3 filtros secuencialmente.
        Usa los parametros aprendidos en fit(), NO re-aprende nada.
        """
        X_out = self.drop_constant.transform(X_data)
        X_out = self.drop_correlated.transform(X_out)
        X_out = self.select_shuffling.transform(X_out)
        return X_out

    def print_summary(self):
        """Imprime un resumen profesional del pipeline de filtrado."""
        print("=" * 60)
        print("RESUMEN DEL PIPELINE DE FILTRADO")
        print("=" * 60)
        print(f"  Features iniciales:    {self.n_features_initial}")
        print(f"  Eliminadas cuasi-constantes:    -{self.n_dropped_constant}")
        print(f"  Eliminadas por correlación:     -{self.n_dropped_correlated} (Spearman)")
        print(f"  Eliminadas por Shuffling (RFI): -{self.n_dropped_shuffling}")
        print(f"  Features seleccionadas finales:  {self.n_features_final}")
        print("=" * 60)
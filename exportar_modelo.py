import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
 
# Conector personalizado a la base de datos
from src.database.connection2 import MySQLConnector
 
def main():
    print("=================================================================")
    print("  ACTIVIDAD FINAL: ENTRENAMIENTO Y PERSISTENCIA DEL MODELO")
    print("=================================================================\n")
 
    os.makedirs('reportes', exist_ok=True)
 
    # -------------------------------------------------------------
    # FASE 1: INGESTA Y PREPARACIÓN DE DATOS (CRISP-DM)
    # -------------------------------------------------------------
    db = MySQLConnector()
 
    query = """
        SELECT 
            p.player_id,
            p.name AS jugador,
            p.sub_position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """
 
    # Nota: 'posicion' usa la sub-posición (Left Winger, Centre-Forward, etc.)
    print("[1] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)
 
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    # Rango de edad permitido en el proyecto: 17 a 45 años
    EDAD_MIN, EDAD_MAX = 17, 45
    total_antes = len(df)
    df = df[df['edad'].between(EDAD_MIN, EDAD_MAX)].copy()
    print(f"[OK] Filtro de edad {EDAD_MIN}-{EDAD_MAX} años: {len(df)} de {total_antes} jugadores")

    # Guardar las medianas de imputación para reutilizar en producción
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = df['estatura_cm'].median()
    mediana_global_edad = df['edad'].median()
 
    # Imputación por mediana agrupada (estatura por posición)
    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)
 
    # Target: Élite (Top 25% de mayor valor comercial)
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)
 
    # Matriz de variables (X) y vector objetivo (y)
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']
 
    # Depuración Anti-NaN
    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]
 
    # División 80/20 Estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    print(f"[OK] Preprocesamiento completado. Registros procesados: {X.shape[0]}.")
    print(f"     Clase 0 (Estándar): {np.sum(y==0)} | Clase 1 (Élite): {np.sum(y==1)}")
    print(f"     Umbral Élite (Percentil 75): {umbral_elite:,.0f} EUR\n")
 
    # -------------------------------------------------------------
    # FASE 2: DIAGNÓSTICO DE OVERFITTING (ÁRBOL LIBRE VS ÁRBOL PODADO)
    # -------------------------------------------------------------
    print("[2] Entrenando Árbol sin restricciones y Árbol con poda previa...")
    tree_libre = DecisionTreeClassifier(random_state=42)
    tree_libre.fit(X_train, y_train)
 
    tree_podado = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
    tree_podado.fit(X_train, y_train)
 
    acc_libre_train = accuracy_score(y_train, tree_libre.predict(X_train))
    acc_libre_test = accuracy_score(y_test, tree_libre.predict(X_test))
    acc_podado_train = accuracy_score(y_train, tree_podado.predict(X_train))
    acc_podado_test = accuracy_score(y_test, tree_podado.predict(X_test))
 
    tabla_overfitting = pd.DataFrame({
        'Modelo': ['Árbol sin restricciones', 'Árbol con poda (max_depth=4)'],
        'Accuracy Train': [acc_libre_train, acc_podado_train],
        'Accuracy Test': [acc_libre_test, acc_podado_test],
    })
    tabla_overfitting['Brecha'] = tabla_overfitting['Accuracy Train'] - tabla_overfitting['Accuracy Test']
    tabla_overfitting.to_csv('reportes/tabla_overfitting.csv', index=False)
 
    print("\n=================================================")
    print("  DIAGNÓSTICO DE OVERFITTING (TRAIN VS TEST)")
    print("=================================================")
    print(tabla_overfitting.round(4).to_string(index=False))
    print(f"\n  • Paradoja de la Exactitud: predecir siempre 'Estándar' ya da Accuracy = {1 - y_test.mean():.4f}")
    print("=================================================\n")
 
    # -------------------------------------------------------------
    # FASE 3: OPTIMIZACIÓN DEL RANDOM FOREST CON GRIDSEARCHCV
    # -------------------------------------------------------------
    print("[3] Ejecutando GridSearchCV sobre Random Forest (Búsqueda sistemática + K-Fold CV)...")
 
    # Definir la Malla (Grid) de combinaciones
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [5, 10, 20],
        'class_weight': ['balanced', None]
    }
 
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # Optimizar buscando el mejor F1-Score
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
 
    print(f"\nMejores Hiperparámetros Encontrados: {grid_search.best_params_}")
    print(f"Mejor F1-Score Promedio en CV: {grid_search.best_score_:.4f}\n")
 
    rf_optimo = grid_search.best_estimator_
 
    # -------------------------------------------------------------
    # FASE 4: EVALUACIÓN FINAL Y CURVA ROC-AUC
    # -------------------------------------------------------------
    y_pred_rf = rf_optimo.predict(X_test)
 
    print("=================================================")
    print("  EVALUACIÓN FINAL DEL RANDOM FOREST (TEST SET)")
    print("=================================================")
    print(classification_report(y_test, y_pred_rf, target_names=['Estándar (0)', 'Élite (1)']))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("=================================================\n")
 
    # Probabilidades para la clase Élite (columna 1)
    y_prob_tree = tree_podado.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_optimo.predict_proba(X_test)[:, 1]
 
    auc_tree = roc_auc_score(y_test, y_prob_tree)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
 
    print("=================================================")
    print("  COMPARATIVA ROC-AUC (CAPACIDAD DE DISCRIMINACIÓN)")
    print("=================================================")
    print(f"  • Árbol de Decisión Único ROC-AUC: {auc_tree:.4f}")
    print(f"  • Random Forest Optimizado ROC-AUC: {auc_rf:.4f}")
    print("=================================================\n")
 
    # Importancia de variables del Random Forest
    importancias = pd.Series(rf_optimo.feature_importances_, index=X.columns).sort_values(ascending=False)
    importancias.to_csv('reportes/importancia_variables.csv', header=['importancia'])
 
    print("Top Variables Más Relevantes según Random Forest:")
    print(importancias.head(5))
    print("\n")
 
    # -------------------------------------------------------------
    # FASE 5: SERIALIZACIÓN DEL MODELO (DEPLOYMENT)
    # -------------------------------------------------------------
    print("[4] Entrenando modelo final con todos los datos y exportando...")
    rf_final = RandomForestClassifier(**grid_search.best_params_, random_state=42, n_jobs=-1)
    rf_final.fit(X, y)
 
    # Crear paquete de despliegue
    paquete_despliegue = {
        'modelo': rf_final,
        'columnas_X': X.columns.tolist(),
        'posiciones_validas': sorted(df['posicion'].unique().tolist()),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite
    }
 
    os.makedirs('models', exist_ok=True)
    ruta_modelo = 'models/modelo_scouting_rf.joblib'
    joblib.dump(paquete_despliegue, ruta_modelo)
    print(f"[OK] Paquete de modelo exportado exitosamente en: '{ruta_modelo}'\n")
 
    # -------------------------------------------------------------
    # FASE 6: VISUALIZACIÓN DE RESULTADOS (GRÁFICAS PARA EL INFORME)
    # -------------------------------------------------------------
    print("[5] Generando gráficas de resultados...")
 
    # Gráfico 1: Diagnóstico de Overfitting (Train vs Test)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    posiciones_x = np.arange(2)
    ancho = 0.35
    ax1.bar(posiciones_x - ancho/2, tabla_overfitting['Accuracy Train'], ancho,
            label='Train', color='#2980b9', edgecolor='black', alpha=0.85)
    ax1.bar(posiciones_x + ancho/2, tabla_overfitting['Accuracy Test'], ancho,
            label='Test', color='#e67e22', edgecolor='black', alpha=0.85)
    ax1.set_xticks(posiciones_x)
    ax1.set_xticklabels(['Sin restricciones', 'Con poda'])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_title('Diagnóstico de Overfitting: Train vs Test', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax1.legend(loc='lower right', frameon=True)
    for i in range(2):
        ax1.text(i - ancho/2, tabla_overfitting['Accuracy Train'][i] + 0.01,
                 f"{tabla_overfitting['Accuracy Train'][i]:.3f}", ha='center')
        ax1.text(i + ancho/2, tabla_overfitting['Accuracy Test'][i] + 0.01,
                 f"{tabla_overfitting['Accuracy Test'][i]:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig('reportes/overfitting_train_vs_test.png', dpi=300)
 
    # Gráfico 2: Curvas ROC y Importancia de Variables
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
 
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
 
    axes[0].plot(fpr_tree, tpr_tree, color='#e67e22', linestyle='--', linewidth=2,
                 label=f'Árbol Único (AUC = {auc_tree:.3f})')
    axes[0].plot(fpr_rf, tpr_rf, color='#27ae60', linewidth=2.5,
                 label=f'Random Forest (AUC = {auc_rf:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Clasificador Aleatorio (AUC = 0.500)')
    axes[0].set_title('Comparativa de Curvas ROC', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Tasa de Falsos Positivos (FPR - Falsas Alarmas)', fontsize=10)
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR - Recall)', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right', frameon=True)
 
    top_importancias = importancias.head(6).sort_values(ascending=True)
    axes[1].barh(top_importancias.index, top_importancias.values, color='#2980b9', edgecolor='black', alpha=0.85)
    axes[1].set_title('Top Importancia de Variables (Random Forest)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Importancia Relativa (Gini Impurity)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6, axis='x')
 
    plt.tight_layout()
    plt.savefig('reportes/evaluacion_rf_vs_tree.png', dpi=300)
    print("[OK] Gráficas guardadas en la carpeta 'reportes/'.")
    plt.show()
 
if __name__ == "__main__":
    main()
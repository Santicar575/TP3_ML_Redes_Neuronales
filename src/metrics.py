import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def predict_classes(model, X):
    """Obtiene las etiquetas predichas (la clase con mayor probabilidad)"""
    y_prob = model.predict(X)
    return np.argmax(y_prob, axis=0)

def accuracy_per_class(y_true, y_pred, num_classes):
    """Calcula el accuracy para cada clase individualmente.
    Retorna un array de tamaño num_classes con el accuracy de cada una."""
    accs = np.zeros(num_classes)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            accs[c] = np.mean(y_pred[mask] == c)
        else:
            accs[c] = 0.0
    return accs

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    Construye la matriz de confusión.
    Las filas son las clases reales y las columnas las predichas.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def f1_score_macro(cm):
    """
    Calcula el F1-Score para cada clase y luego devuelve el promedio (Macro).
    """
    num_classes = cm.shape[0]
    f1_scores = []
    
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        f1_scores.append(f1)
        
    return np.mean(f1_scores)

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Genera un heatmap visual de la matriz de confusión"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title(title)
    plt.ylabel('Real Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_accuracy_boxplot(class_accs, title="Accuracy por Clase"):
    """Genera un boxplot con los accuracies por clase."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=class_accs, orient='v', color='skyblue', width=0.4)
    sns.stripplot(data=class_accs, orient='v', color='darkblue', alpha=0.5, size=4)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X, y_one_hot, y_labels, num_classes, name_set):
    """
    Calcula y printea todas las métricas, incluyendo accuracy por clase
    y un boxplot de los mismos.
    """
    y_pred_labels = predict_classes(model, X)
    y_prob = model.predict(X) 
    loss = model.cross_entropy(y_one_hot, y_prob)
    
    class_accs = accuracy_per_class(y_labels, y_pred_labels, num_classes)
    cm = compute_confusion_matrix(y_labels, y_pred_labels, num_classes)
    f1_macro = f1_score_macro(cm)
    
    print(f"--- Resultados para el Conjunto de {name_set} ---")
    print(f"Cross-Entropy Loss : {loss:.4f}")
    print(f"F1-Score (Macro)   : {f1_macro:.4f}\n")
    
    # Tabla de Accuracy por Clase
    header = f"{'Clase':>7} | {'Acc (%)':>8} | {'Muestras':>8}"
    sep    = "-" * len(header)
    rows = []
    for c in range(num_classes):
        n = int(np.sum(y_labels == c))
        rows.append(f"{c:>7} | {class_accs[c]*100:>7.2f}% | {n:>8}")
    
    n_rows = len(rows)
    third = (n_rows + 2) // 3
    cols = [rows[i*third:(i+1)*third] for i in range(3)]
    
    max_len = max(len(col) for col in cols)
    for col in cols:
        while len(col) < max_len:
            col.append(" " * len(header))
    
    spacer = "    │    "
    print(f"Accuracy por Clase - {name_set}")
    print(spacer.join([header] * 3))
    print(spacer.join([sep] * 3))
    for row_parts in zip(*cols):
        print(spacer.join(row_parts))
    print(spacer.join([sep] * 3))
    avg_acc = np.mean(class_accs)
    print(f"Accuracy promedio: {avg_acc:.4f} ({avg_acc*100:.2f}%)  —  Total muestras: {len(y_labels)}\n")
    
    plot_accuracy_boxplot(class_accs, title=f"Accuracy por Clase - {name_set}")
    
    return cm, class_accs
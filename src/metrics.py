import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def predict_classes(model, X):
    """Obtiene las etiquetas predichas (la clase con mayor probabilidad)"""
    y_prob = model.predict(X)
    return np.argmax(y_prob, axis=0)

def accuracy_score(y_true, y_pred):
    """Calcula el porcentaje total de aciertos"""
    return np.mean(y_true == y_pred)

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
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title(title)
    plt.ylabel('Real Label')
    plt.xlabel('Predicted Label')
    plt.show()

def evaluate_model(model, X, y_one_hot, y_labels, num_classes, name_set):
    """
    Calcula y printea todas las métricas
    """
    y_pred_labels = predict_classes(model, X)
    y_prob = model.predict(X) 
    loss = model.cross_entropy(y_one_hot, y_prob)
    
    acc = accuracy_score(y_labels, y_pred_labels)
    cm = compute_confusion_matrix(y_labels, y_pred_labels, num_classes)
    f1_macro = f1_score_macro(cm)
    
    print(f"--- Resultados para el Conjunto de {name_set} ---")
    print(f"Cross-Entropy Loss : {loss:.4f}")
    print(f"Accuracy           : {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score (Macro)   : {f1_macro:.4f}\n")
    
    return cm
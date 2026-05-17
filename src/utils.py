import numpy as np
import json
import os
from tqdm import tqdm
from src.MLP import MLP
from src.metrics import predict_classes, compute_confusion_matrix, f1_score_macro
from src.MLP_Torch import MLP_Torch, Trainer
from itertools import product
import torch
import random
from torch.utils.data import TensorDataset, DataLoader


def cross_val_mlp(X, y_labels, num_classes, n_layers, nodes_per_layer,
                  k=5, random_state=1973, framework="numpy", **fit_kwargs):
    """

    Args:
        X (np.ndarray): Matriz de features, shape (n_samples, n_features).
        y_labels (np.ndarray): Vector de etiquetas enteras
        num_classes (int): Cantidad de clases.
        n_layers (int): Cantidad de capas del MLP (sin contar entrada).
        nodes_per_layer (list[int]): Nodos por capa incluyendo entrada y salida.
        k (int): Número de folds.
        random_state (int): Semilla para reproducibilidad.
        framework (str): Framework a usar ("numpy" o "pytorch")
        **fit_kwargs: Argumentos para MLP.fit() (eta_0, epochs, optimizer,
                      batch_size, lr_schedule, lambda_l2, etc.)
    Returns:
        float: F1 macro promedio de validation sobre los k folds.
    """

    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    f1_val_scores = []

    all_params = {'nodes_per_layer': nodes_per_layer, **fit_kwargs}
    batch_size = all_params.get('batch_size', 128)

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train_fold = X[train_idx]
        y_train_fold = y_labels[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y_labels[val_idx]
        
        if framework == "numpy":
            X_train_t = X_train_fold.T
            X_val_t = X_val_fold.T

            # One-hot encoding
            y_train_oh = np.zeros((num_classes, len(y_train_fold)))
            for idx in range(len(y_train_fold)):
                y_train_oh[int(y_train_fold[idx]), idx] = 1

            y_val_oh = np.zeros((num_classes, len(y_val_fold)))
            for idx in range(len(y_val_fold)):
                y_val_oh[int(y_val_fold[idx]), idx] = 1

            # Crear y entrenar modelo
            model = MLP(n_layers=n_layers,
                        nodes_per_layer=nodes_per_layer,
                        random_seed=random_state + i)

            model.fit(X_train_t, y_train_oh,
                    X_val=X_val_t, y_val=y_val_oh,
                    **fit_kwargs)

            y_pred_val = predict_classes(model, X_val_t)
            y_val_fold_real = y_val_fold
        
        elif framework == "pytorch":

            # Transformo los datos a tensores de PyTorch
            X_train_t = torch.tensor(X_train_fold, dtype=torch.float32)
            y_train_t = torch.tensor(y_train_fold, dtype=torch.long)
            X_val_t = torch.tensor(X_val_fold, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_fold, dtype=torch.long)

            train_ds = TensorDataset(X_train_t, y_train_t)
            val_ds = TensorDataset(X_val_t, y_val_t)

            generator = torch.Generator().manual_seed(random_state + i)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

            # Instanciar Modelo y Entrenar
            torch.manual_seed(random_state + i) 
            model = MLP_Torch(all_params)
            trainer = Trainer(all_params, model, train_loader, val_loader, do_prints=False)
            trainer.train()

            # Pongo el modelo en evaluation mode para no usar dropout ni gradientes
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(trainer.device)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(targets.cpu().numpy())
            
            y_pred_val = np.array(all_preds)
            y_val_fold_real = np.array(all_targets)

        cm_val = compute_confusion_matrix(y_val_fold_real, y_pred_val, num_classes)
        f1_val = f1_score_macro(cm_val)
        f1_val_scores.append(f1_val)

    mean_f1 = np.mean(f1_val_scores)

    return mean_f1


def grid_search(X, y_labels, num_classes, param_grid,
                    k=5, random_state=1973, config_name=None, framework="numpy", **fixed_fit_kwargs):
    """
    Grid search con k-fold cross validation.
    Prueba todas las combinaciones de hiperparámetros del param_grid
    y devuelve la mejor configuración según F1 macro de validation.

    La arquitectura se puede variar incluyendo 'nodes_per_layer' en
    param_grid. n_layers se calcula automáticamente.

    Args:
        X (np.ndarray): Matriz de features
        y_labels (np.ndarray): labels
        num_classes (int): Cantidad de clases
        param_grid (dict): Diccionario donde cada clave es un hiperparámetro
                           y cada valor es una lista de opciones a probar
        k (int): Número de folds para CV
        random_state (int): Semilla para reproducibilidad
        framework (str): Framework a usar ("numpy" o "pytorch")
        **fixed_fit_kwargs: Hiperparámetros fijos que no se buscan
    Returns:
        tuple: (best_params, results)
            - best_params (dict): Hiperparámetros con mejor F1 macro val
            - results (list[dict]): Todas las combinaciones evaluadas ordenadas de mayor a menor F1
    """

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    total = len(combinations)

    results = []
    best_f1 = 0.0

    pbar = tqdm(combinations, desc="Grid Search", unit="combo")
    for combo in pbar:
        params = dict(zip(param_names, combo))

        # Extraer arquitectura de los params
        nodes = params.pop('nodes_per_layer')
        n_layers = len(nodes) - 1

        # Los params restantes van a MLP.fit()
        fit_kwargs = {**fixed_fit_kwargs, **params}

        pbar.set_postfix({'best_f1': f'{best_f1:.4f}', 'arch': str(nodes)})

        f1_val = cross_val_mlp(
            X=X,
            y_labels=y_labels,
            num_classes=num_classes,
            n_layers=n_layers,
            nodes_per_layer=nodes,
            k=k,
            random_state=random_state,
            framework=framework,
            **fit_kwargs
        )

        # Guardar resultado con la arquitectura incluida
        full_params = {'nodes_per_layer': nodes, **params}
        results.append({'params': full_params, 'f1_val': f1_val})

        if f1_val > best_f1:
            best_f1 = f1_val

    # Ordenar de mejor a peor
    results.sort(key=lambda r: r['f1_val'], reverse=True)

    best = results[0]
    print(f"\nMejor configuración: {best['params']}")
    print(f"F1 macro val: {best['f1_val']:.4f}")

    # Guardar mejores parámetros en configs/
    if config_name is not None:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
        os.makedirs(config_dir, exist_ok=True)

        # Convertir tipos numpy a nativos de Python para JSON
        def to_native(v):
            if isinstance(v, (list, tuple)):
                return [to_native(x) for x in v]
            elif isinstance(v, np.integer):
                return int(v)
            elif isinstance(v, np.floating):
                return float(v)
            return v

        # Unificar todos los parámetros en un solo dict
        all_params = {**best['params'], **fixed_fit_kwargs}
        best_params_serializable = {k: to_native(v) for k, v in all_params.items()}

        config_data = {
            'best_params': best_params_serializable,
            'f1_macro_val': round(float(best['f1_val']), 4),
        }

        config_path = os.path.join(config_dir, f'{config_name}.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)

    return best['params'], results


def set_seed(seed=1973):
    """Fija todas las semillas para asegurar reproducibilidad en PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Forzar a PyTorch a usar algoritmos determinísticos
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
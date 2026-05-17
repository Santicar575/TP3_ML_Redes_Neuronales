import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import copy

class ImageDataset(Dataset):
    def __init__(self, labels_dir, img_dir, transform=None, target_transform=None):
        self.img_labels = np.load(labels_dir)
        self.img_dir = np.load(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_dir[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_dataloaders(labels_dir, img_dir, params, transform, target_transform, val_fraction=0.2, random_seed=1973):
    dataset = ImageDataset(labels_dir, img_dir, transform, target_transform)
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    batch_size = params.get('batch_size', 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader

class MLP_Torch(nn.Module):
    def __init__(self, params):
        super().__init__()
        nodes = params['nodes_per_layer']
        input_dim = nodes[0]
        hidden_dims = nodes[1:-1]
        n_classes = nodes[-1]
        layers = []
        in_dim = input_dim
        dropout_p = params.get('dropout_p', 0.0)
        act_name = params.get('activation_func', 'relu').lower()

        def get_activation(name):
            if name == 'leakyrelu': return nn.LeakyReLU()
            elif name == 'silu' or name == 'swish': return nn.SiLU()
            elif name == 'gelu': return nn.GELU()
            return nn.ReLU()
        
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_activation(act_name))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def predict(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs

class Trainer():
    def __init__(self, params, model, train_loader, val_loader, do_prints=True):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = self.get_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.do_prints = do_prints

    def get_optimizer(self):
        opt_name = self.params.get("optimizer", "adam").lower()
        lr = self.params.get("eta_0", 0.001)
        
        # Weight Decay
        wd = self.params.get("lambda_l2", 0.0)

        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Optimizer {opt_name} not supported.")

    def get_scheduler(self):
        
        schedule_type = self.params.get("lr_schedule", None)
        
        if schedule_type == "lineal":
            K = self.params.get("K", 150)
            eta_K = self.params.get("eta_K", 0.0001)
            eta_0 = self.params.get("eta_0", 0.005)

            # Funcion lambda que decrementa linealmente el learning rate
            lr_lambda = lambda epoch: ((1 - epoch/K) * eta_0 + (epoch/K) * eta_K) / eta_0 if epoch < K else eta_K / eta_0
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
        elif schedule_type == "exponencial":
            c = self.params.get("c", 0.90)
            s = self.params.get("s", 20.0)
            gamma = c ** (1/s) 
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
            
        return None
    
    def train(self):
        early_stopping = self.params.get("early_stopping", False)
        patience = self.params.get("patience", 10)
        min_delta = self.params.get("min_delta", 0.001)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        loss_hist = []
        val_loss_hist = []

        for epoch in range(self.params.get("epochs", 300)):
            train_loss, val_loss = self.run_epoch()
            loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            if self.scheduler:
                self.scheduler.step()

            if early_stopping:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.do_prints:
                            print(f"Early Stopping en epoch {epoch+1}!")
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        break
                        
        return loss_hist, val_loss_hist

    def run_epoch(self):
        self.model.train()
        total_loss_train = 0

        # training loop
        for batch in self.train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient Clipping
            clip_norm = self.params.get("clip_norm", 0.0)
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            
            self.optimizer.step()
            total_loss_train += loss.item()
        
        self.model.eval()
        total_loss_val = 0
        
        # validation loop
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss_val += loss.item()
        train_loss = total_loss_train / len(self.train_loader)
        val_loss = total_loss_val / len(self.val_loader)
        return train_loss, val_loss

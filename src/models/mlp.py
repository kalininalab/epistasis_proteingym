import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from scipy.stats import spearmanr
from src.constants import set_seed
from src.utils import convert_name_to_gfp, shift_mutation_positions_up


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def train_mlp_model(x_train, y_train, device, batch_size=64, lr=1e-3, epochs=500, patience=10):
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    model = MLP(x_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    counter = 0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        loss_history.append(avg)

        # if epoch % 10 == 0:
            # print(f"Epoch {epoch} | Loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print(f"Early stopping at epoch {epoch}")
                break

    return model


def evaluate_on_test_sets(model, x_test, y_test, device="cpu", log_target=True):
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(x_tensor).squeeze().cpu().numpy()

    # Inverse transform if log was used
    if log_target:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    # Spearman correlation
    rho = spearmanr(y_pred, y_test.flatten())[0]

    return rho


def mlp(result_all, model_dir, epi_dir, dataset_name, seeds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load training data
    input_dir = model_dir / dataset_name
    for file_path in input_dir.glob("*.csv"): # iterate over datasets
        df = pd.read_csv(file_path)
        df["num_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
        singles = df[df['num_mutations'] == 1].copy()
        multis = df[df['num_mutations'] > 1].copy()
        dataset = file_path.stem
        
        if dataset_name == "somermeyer":
            selected = pd.read_csv(epi_dir / f"{convert_name_to_gfp(dataset)}.csv")
            selected.rename(columns={"aa_genotype": "mutant"}, inplace=True)
            selected['mutant'] = selected['mutant'].apply(shift_mutation_positions_up)
        else:
            selected = pd.read_csv(epi_dir / file_path.name)
        
        selected = selected[['mutant', 'epistatic']]
        multis = multis.merge(selected, on='mutant', how='left')
        epistatic = multis[multis["epistatic"].notna() & multis["epistatic"]]
        
        if len(epistatic) == 0: # no epistasis detected for the dataset
            continue

        seqs_train = singles['mutated_sequence'].apply(lambda x: x.rstrip('*')).tolist()
        y_train = np.array(singles['DMS_score'])

        seqs_epistatic = epistatic['mutated_sequence'].apply(lambda x: x.rstrip('*')).tolist()
        y_test_epistatic = np.array(epistatic['DMS_score'])

        spearman_all = []
        spearman_epistatic = []
        for seed in range(seeds):
            # sample as the same size as epistatic points
            set_seed(seed)
            sample = multis.sample(n=len(epistatic), random_state=seed)
            seqs_all = sample['mutated_sequence'].apply(lambda x: x.rstrip('*')).tolist()
            y_test_all = np.array(sample['DMS_score'])

            enc = preprocessing.OneHotEncoder(handle_unknown="ignore")
            max_length = max(max(len(seq) for seq in seqs_train),
                        max(len(seq) for seq in seqs_epistatic),
                        max(len(seq) for seq in seqs_all))
            seqs_train = [list(seq) for seq in seqs_train]
            padded_sequences = [seq + [''] * (max_length - len(seq)) for seq in seqs_train]
            padded_array = np.array(padded_sequences)
            enc.fit(padded_array)
            x_train = enc.transform(padded_array).toarray()

            test_sequences = [list(seq) for seq in seqs_all]
            padded_sequences = [seq + [''] * (max_length - len(seq)) for seq in test_sequences]
            padded_array = np.array(padded_sequences)
            genotype_matrix = enc.transform(padded_array)
            x_test_all = genotype_matrix.toarray()

            test_sequences = [list(seq) for seq in seqs_epistatic]
            padded_sequences = [seq + [''] * (max_length - len(seq)) for seq in test_sequences]
            padded_array = np.array(padded_sequences)
            genotype_matrix = enc.transform(padded_array)
            x_test_epistatic = genotype_matrix.toarray()
            
            if dataset_name == "somermeyer":
                # Log-transform the target variable
                y_train = np.log1p(y_train)
                y_test_all = np.log1p(y_test_all)
                y_test_epistatic = np.log1p(y_test_epistatic)

            # Train MLP model
            model = train_mlp_model(x_train, y_train, device)

            # Evaluate
            log_target = True if dataset_name == "somermeyer" else False
            
            spearman_all.append(evaluate_on_test_sets(
                model,
                x_test_all,
                y_test_all,
                device=device,
                log_target=log_target
            ))

            spearman_epistatic.append(evaluate_on_test_sets(
                    model,
                    x_test_epistatic,
                    y_test_epistatic,
                    device=device,
                    log_target=log_target
            ))
            
        result_all.loc["MLP", dataset + '_all'] = f"{np.mean(spearman_all):.2f}"
        result_all.loc["MLP", dataset + '_epistatic'] = f"{np.mean(spearman_epistatic):.2f}"

    return result_all
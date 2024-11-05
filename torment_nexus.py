import os
import angr
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split 
from typing import Tuple
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report  # Added import for evaluation

class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(384, 128),  # Changed input dimension to match embedding size and output to 128
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128)
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128)
            )
        )
        self.conv3 = GINConv(  # Added an additional convolutional layer
            torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128)
            )
        )
        self.fc1 = torch.nn.Linear(128, 64)  # Adjusted dimensions
        self.fc2 = torch.nn.Linear(64, 2)  # Output remains 2 for binary classification

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # Added batch extraction
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # Added an additional layer
        x = F.relu(x)

        x = global_add_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class TormentNexus:
    def preprocess_binary(binary_file:str) -> Data:
        if binary_file.endswith('.exe'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device)
            
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embedding_model = embedding_model.to(device) 
            print(f"Processing executable {binary_file}")
            
            angr_project = angr.Project(binary_file, auto_load_libs=False)
            cfg = angr_project.analyses.CFGFast()

            functions = list(angr_project.kb.functions.values())
            function_addr_to_index = {function.addr: idx for idx, function in enumerate(functions)}

            nodes = []
            for function in functions:
                instructions = []
                for block in function.blocks:
                    capstone_block = block.capstone
                    for insn in capstone_block.insns:
                        instructions.append(insn.mnemonic)
                instruction_sequence = ' '.join(instructions)

                embedding = embedding_model.encode(instruction_sequence)
                nodes.append(embedding)

            edge_index = []
            callgraph = angr_project.kb.callgraph
            for src_addr, dst_addr in callgraph.edges():
                src_idx = function_addr_to_index.get(src_addr)
                dst_idx = function_addr_to_index.get(dst_addr)
                if src_idx is not None and dst_idx is not None:
                    edge_index.append([src_idx, dst_idx])

            node_embeddings = torch.tensor(nodes, dtype=torch.float)
            if len(edge_index) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            data = Data(x=node_embeddings, edge_index=edge_index, dtype=torch.long)
            return data
        else:
            print("File found, but is not executable.")

    def generate_data_list_from_binaries(binary_dir:str = './binaries', ) -> list[Data]:

        data_list: list[Data] = []

        for binary_file in os.listdir(f"{binary_dir}/malware"):
            try:
                data = TormentNexus.preprocess_binary(f"{binary_dir}/malware/{binary_file}")
                data.y = 1
                data_list.append(data)
            except:
                ''

        for binary_file in os.listdir(f"{binary_dir}/benign"):
            try:
                data = TormentNexus.preprocess_binary(f"{binary_dir}/benign/{binary_file}")
                data.y = 0
                data_list.append(data)
            except:
                ''
            
        
        return data_list

    def get_train_test_split(data_list: list[Data]) -> Tuple[list[Data], list[Data]]:
        train_data, test_data = train_test_split(
            data_list, test_size=0.2, random_state=42,
            stratify=[d.y for d in data_list]
        )
        return train_data, test_data

    def get_model() -> GIN:
        model = GIN()
        return model

    def train_model(model: GIN, train_data: list[Data], test_data: list[Data]):

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        epochs = 10

        for epoch in range(epochs):
            model.train()  
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = loss_fn(output, batch.y)
                loss.backward()
                optimizer.step()
        
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    output = model(batch)
                    preds = output.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
            print(f"Epoch {epoch+1}/{epochs}")
            print(classification_report(all_labels, all_preds)) 

    def save_model_weights(model: GIN, model_path: str = './gin_model_weights.pth'):
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")
        
    def classify_binary(model: GIN = None, binary_file_path: str = './binaries/malware/game.exe') -> bool:

        if model == None:
            model = GIN()
            model.load_state_dict(torch.load('./gin_model_weights.pth')) 
            model.eval()

        new_data = TormentNexus.preprocess_binary(binary_file_path)

        new_data.batch = torch.zeros(new_data.x.size(0), dtype=torch.long)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        new_data = new_data.to(device)

        with torch.no_grad():
            output = model(new_data)
            prediction = output.argmax(dim=1).item()
            # print(f"Prediction for the binary: {'Malware' if prediction == 1 else 'Benign'}")
        
        return prediction == 1

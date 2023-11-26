import argparse
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import gzip
import os
from torch_geometric.nn import TopKPooling, GATConv, SAGEConv, GCNConv
import torch
import pandas as pd
from torch_geometric.data import Data

import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, SAGPooling
import copy
import torch
from matplotlib import pyplot as plotting_my_graph_object

def get_node_feature_dims():
    return [119, 5, 12, 12, 10, 6, 6, 2, 2]


def get_edge_feature_dims():
    return [5, 6, 2]


class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_embedding_list = torch.nn.ModuleList()
        full_node_feature_dims = get_node_feature_dims()
        for i, dim in enumerate(full_node_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.node_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.node_embedding_list[i](x[:,i])
        return x_embedding


class EdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(EdgeEncoder, self).__init__()
        full_edge_feature_dims = get_edge_feature_dims()
        self.edge_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_edge_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.edge_embedding_list.append(emb)

    def forward(self, edge_attr):
        edge_embedding = 0
        for i in range(edge_attr.shape[1]):
            edge_embedding += self.edge_embedding_list[i](edge_attr[:,i])
        return edge_embedding


def load_csv(file_path):
    with gzip.open(file_path, 'rt') as file:
        df = pd.read_csv(file,header=None)
    return df

# Load graph labels, num_nodes, and num_edges
# %cd /content/drive/MyDrive/Data mining/dataset/dataset_2/train

# print(graph_labels)
# print(num_nodes)
# print(num_edges)
# print(node_features)
# print(edges)
# print(edge_features)


# %cd /content/drive/MyDrive/Data mining/dataset/dataset_2/valid

# print(graph_labels1)



# Your load_csv function remains the same

def create_pyg_data(num_nodes,num_edges,node_features,edges,edge_features):
    # Extract graph information for the given index
    dataset = []
    number_of_ones = 0
    number_of_graphs = 0
    nl =0
    el = 0
    for graph_idx in range(len(num_nodes)):
        # label = graph_labels.iloc[graph_idx].values[0]
        number_of_nodes = num_nodes.iloc[graph_idx].values[0]
        number_of_edges = num_edges.iloc[graph_idx].values[0]
        edge_fe = edge_features.iloc[el:(el+number_of_edges)]
        ed = edges.iloc[el:(el + number_of_edges)]
        edge_fe = edge_fe.values.tolist()
        node_fe = node_features.iloc[nl:(nl + number_of_nodes)]
        ed = ed.values.tolist()
        edge_fe = torch.tensor(edge_fe, dtype = torch.long)
        node_fe = node_fe.values.tolist()
        ed = torch.LongTensor(ed)
        node_fe = torch.tensor(node_fe, dtype = torch.long)
        nl+=number_of_nodes
        el += number_of_edges
        # if(np.isnan(label)):
        #     continue

        if(len(ed)!=0):ed = ed.T
        # label = torch.Tensor([label])
        
        data = Data(edge_attr=edge_fe,x=node_fe,edge_index=ed)
        number_of_graphs+=1
        # number_of_ones+=label
        dataset.append(data)
    return number_of_graphs,number_of_ones,dataset
        # train_dataset.append(dataset.get(i,graph_labels,num_nodes,num_edges,node_features,edges,edge_features))





    
    # # Handle NaN values in the label column
    # label = 0.0 if pd.isna(label) else label

    # # Convert label to numpy.float64 (if it's not already)
    # label = np.float64(label)
    # nodes = num_nodes.iloc[graph_idx].values[0]
    # edges_count = num_edges.iloc[graph_idx].values[0]


    # # Calculate the start and end index for the nodes of the current graph
    # start_idx = num_nodes.iloc[:graph_idx].sum().values[0] if graph_idx > 0 else 0
    # end_idx = start_idx + nodes

    # # Extract node features for the current graph
    # node_features_graph = node_features.iloc[start_idx:end_idx]

    # # Calculate the start and end index for the edges of the current graph
    # start_edge_idx = num_edges.iloc[:graph_idx].sum().values[0] if graph_idx > 0 else 0
    # end_edge_idx = start_edge_idx + edges_count

    # # Extract edges for the current graph
    # edges_graph = edges.iloc[start_edge_idx:end_edge_idx]

    # # Extract edge features for the current graph
    # edge_features_graph = edge_features.iloc[start_edge_idx:end_edge_idx]

    # # Convert node features, edges, and edge features to PyTorch tensors
    # x = torch.tensor(node_features_graph.values, dtype=torch.float)
    # edge_index = torch.tensor(edges_graph.values, dtype=torch.long).t().contiguous()
    # edge_attr = torch.tensor(edge_features_graph.values, dtype=torch.float)
    # y = torch.tensor([label], dtype=torch.float)
    # # Create a PyTorch Geometric Data object
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # return data

# Example: Create a PyTorch Geometric Data object for the first graph
# graph_index = 1
# graph_data = create_pyg_data(graph_index)
# print(graph_data)




# Your load_csv function remains the same

# def create_pyg_data1(graph_idx):
#     # Extract graph information for the given index
#     label = graph_labels1.iloc[graph_idx].values[0]
#     # Handle NaN values in the label column
#     label = 0.0 if pd.isna(label) else label

#     # Convert label to numpy.float64 (if it's not already)
#     label = np.float64(label)
#     nodes = num_nodes1.iloc[graph_idx].values[0]
#     edges_count = num_edges1.iloc[graph_idx].values[0]

#     # Calculate the start and end index for the nodes of the current graph
#     start_idx = num_nodes1.iloc[:graph_idx].sum().values[0] if graph_idx > 0 else 0
#     end_idx = start_idx + nodes

#     # Extract node features for the current graph
#     node_features_graph = node_features1.iloc[start_idx:end_idx]

#     # Calculate the start and end index for the edges of the current graph
#     start_edge_idx = num_edges1.iloc[:graph_idx].sum().values[0] if graph_idx > 0 else 0
#     end_edge_idx = start_edge_idx + edges_count

#     # Extract edges for the current graph
#     edges_graph = edges1.iloc[start_edge_idx:end_edge_idx]

#     # Extract edge features for the current graph
#     edge_features_graph = edge_features1.iloc[start_edge_idx:end_edge_idx]

#     # Convert node features, edges, and edge features to PyTorch tensors
#     x = torch.tensor(node_features_graph.values, dtype=torch.float)
#     edge_index = torch.tensor(edges_graph.values, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(edge_features_graph.values, dtype=torch.float)
#     y = torch.tensor([label], dtype=torch.float)
#     # Create a PyTorch Geometric Data object
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

#     return data

# Example: Create a PyTorch Geometric Data object for the first graph
# graph_index = 1
# graph_data = create_pyg_data1(graph_index)
# print(graph_data)



class CustomDataset(Dataset):
    def __init__(self,graph_labels, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)

        # Assuming you have the number of graphs available
        self.num_graphs = len(graph_labels)

    def len(self):
        return self.num_graphs

    def get(self, idx,graph_labels,num_nodes,num_edges,node_features,edges,edge_features):
        # Use the create_pyg_data function to get PyTorch Geometric Data object
        return create_pyg_data(idx,graph_labels,num_nodes,num_edges,node_features,edges,edge_features)

# Create an instance of the CustomDataset



# class CustomDataset1(Dataset):
#     def __init__(self,graph_labels1, root, transform=None, pre_transform=None):
#         super(CustomDataset1, self).__init__(root, transform, pre_transform)

#         # Assuming you have the number of graphs available
#         self.num_graphs = len(graph_labels1)

#     def len(self):
#         return self.num_graphs

#     def get(self, idx):
#         # Use the create_pyg_data function to get PyTorch Geometric Data object
#         return create_pyg_data1(idx,)

# Create an instance of the CustomDataset











 


class GATModel(torch.nn.Module):
    def __init__(self):
        super(GATModel, self).__init__()
        # torch.manual_seed(12345)
        hidden_channels = 32
        self.n_encod = NodeEncoder(16)
        self.e_encod = EdgeEncoder(2)
        self.conv1 = GATConv(16, hidden_channels, edge_dim=2)
        self.conv2 = GATConv(hidden_channels, hidden_channels,edge_dim= 2)
        self.conv3 = GATConv(hidden_channels, hidden_channels,edge_dim= 2)
        self.lin = torch.nn.Linear(hidden_channels*3, 1)
        # self.lin = torch.nn.Sequential(
        #     Linear(3 * hidden_channels, hidden_channels),
        #     torch.nn.ReLU(),
        #     Linear(hidden_channels, 1)
        # )
        # torch.nn.init.xavier_uniform_(self.lin[0].weight,gain=1.414)


    def forward(self, X, edge_index, batch,Ed_f):

        out1 = self.conv1(self.n_encod(X), edge_index, edge_attr = self.e_encod(Ed_f))
        out2 = self.conv2(out1, edge_index, edge_attr = self.e_encod(Ed_f))
        out3 = self.conv3(out2, edge_index, edge_attr = self.e_encod(Ed_f))


        xlin = self.lin(gap(torch.cat((out1, out2, out3), 1), batch))

        return xlin
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = [global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)]  # [batch_size, hidden_channels]
        x = torch.cat(x, dim = 1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.lin(x)
        x = F.sigmoid(x).squeeze(1)
        
        # if(x[0] < 0.5)
        # le = 0
        # while le<30:
        #   print(x[le])
        #   le+=1
        # print(torch.sum(x))
        return x






# from IPython.display import Javascript
# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))



def train(model,train_loader,optimizer,criterion):
    # created the trainikng model
    model.train()
    # print('hello')
    tot = 0
    num = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
         optimizer.zero_grad()  # Clear gradients.
         out = model(data.x, data.edge_index, data.batch,data.edge_attr).float().flatten()  # Perform a single forward pass.
        #  print(out)
        #  y = torch.reshape(data.y, (-1, 1))
         loss = criterion(out, data.y)  # Compute the loss.
        #  print(loss)
         tot +=  loss.item()*data.num_graphs
         num += data.num_graphs
         loss.backward()  # Derive gradients.
        #  torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
         optimizer.step()  # Update parameters based on gradients.
    tot #/= num
    print(tot)
    return tot


def auroc_loss(y_true, y_scores):
    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_scores)
    # Use 1 - AUROC as the loss
    loss = 1 - auroc
    return loss




def test(model,loader,criterion):
    #created the modewl
    model.eval()
    # model.eval()
    with torch.no_grad():
        # predictions = []
        y_true = []
        y_scores = []
        #The below is the loop for loader to test
        for data in loader:
            out = model(data.x,data.edge_index,data.batch,data.edge_attr)
            label = data.y
            out = out.float()
            y_true.extend(label.numpy())
            out = out.flatten()
            y_scores.extend(out.numpy())
            #Here are values
        y_true = torch.Tensor(y_true)
        y_scores = torch.Tensor(y_scores)
        error = mean_squared_error(y_true, y_scores,squared=False)
        #returning the answer
        return error

    # y_true = []
    # y_scores = []

    # for data in loader:  # Iterate in batches over the training/test dataset.
    #     out = model(data.x, data.edge_index, data.batch,data.edge_attr).float()
    #     pred_probs = out.detach().cpu().numpy()

    #     y_true.extend(data.y.cpu().numpy())
    #     y_scores.extend(pred_probs)

    # y_true = np.array(y_true)
    # y_scores = np.array(y_scores)

    # # Calculate ROC-AUC score
    # roc_auc = roc_auc_score(y_true, y_scores)

    # # Convert predicted probabilities to binary predictions using a threshold (e.g., 0.5)
    # binary_preds = (y_scores >= 0.5).astype(int)

    # # Calculate accuracy
    # correct = np.sum(binary_preds == y_true)
    # accuracy = correct / len(loader.dataset)

    # return accuracy, roc_auc


# def test(loader):
#      model.eval()

#      correct = 0
#      for data in loader:  # Iterate in batches over the training/test dataset.
#          out = model(data.x, data.edge_index, data.batch)
#          pred = out >= 0.5  # Use the class with highest probability.

#          correct += int((pred == data.y).sum())  # Check against ground-truth labels.
#      return correct / len(loader.dataset)  # Derive ratio of correct predictions.









def tocsv(y_arr, *, task):
    r"""Writes the numpy array to a csv file.
    params:
        y_arr: np.ndarray. A vector of all the predictions. Classes for
        classification and the regression value predicted for regression.

        task: str. Must be either of "classification" or "regression".
        Must be a keyword argument.
    Outputs a file named "y_classification.csv" or "y_regression.csv" in
    the directory it is called from. Must only be run once. In case outputs
    are generated from batches, only call this output on all the predictions
    from all the batches collected in a single numpy array. This means it'll
    only be called once.

    This code ensures this by checking if the file already exists, and does
    not over-write the csv files. It just raises an error.

    Finally, do not shuffle the test dataset as then matching the outputs
    will not work.
    """
    import os
    import numpy as np
    import pandas as pd
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)


# The following is just for an example to show how to create the
# right data format, and how to call the function.
#
# The following assumes binary classification task. And the final
# output of the model is a number between 0 and 1, the probability
# of a sample belonging to class 1. For ROC-AUC, we need to save
# this number to the csv file.
#
# Notice that the example code for regression will also be the same.
def test(model, test_loader, device):
    model.eval()
    all_ys = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            ys_this_batch = output.cpu().numpy().tolist()
            all_ys.extend(ys_this_batch)
    numpy_ys = np.asarray(all_ys)
    tocsv(numpy_ys, task="regression") # <- Called outside the loop. Called in the eval code.







def main():
    parser = argparse.ArgumentParser(description="Evaluating the classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    print(f"Evaluating the classification model. Model will be loaded from {args.model_path}. Test dataset will be loaded from {args.dataset_path}.")

    # graph_labels = load_csv(os.path.join(args.dataset_path,'graph_labels.csv.gz'))
    num_nodes = load_csv(os.path.join(args.dataset_path,'num_nodes.csv.gz'))
    num_edges = load_csv(os.path.join(args.dataset_path,'num_edges.csv.gz'))
    node_features = load_csv(os.path.join(args.dataset_path,'node_features.csv.gz'))
    edges = load_csv(os.path.join(args.dataset_path,'edges.csv.gz'))
    edge_features = load_csv(os.path.join(args.dataset_path,'edge_features.csv.gz'))
    number_of_graphs,number_of_ones,train_dataset = create_pyg_data(num_nodes,num_edges,node_features,edges,edge_features)

    # dataset = CustomDataset(graph_labels,root='./')
    # print(type(dataset))
    # print(dataset.get(0))
    # train_dataset = []
    # for i in range(dataset.len()):
    #     train_dataset.append(dataset.get(i,graph_labels,num_nodes,num_edges,node_features,edges,edge_features))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    model = GATModel()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    model.eval()
    # model.eval()
    with torch.no_grad():
        # predictions = []
        y_true = []
        y_scores = []
        #The below is the loop for loader to test
        for data in train_loader:
            out = model(data.x,data.edge_index,data.batch,data.edge_attr)
            # label = data.y
            out = out.float()
            # y_true.extend(label.numpy())
            out = out.flatten()
            y_scores.extend(out.numpy())
            #Here are values
    y_scores = np.asarray(y_scores)
    # tocsv(y_scores, task="classification") # <- Called outside the loop. Called in the eval code.
    tocsv(y_scores, task="regression") # <- Called outside the loop. Called in the eval code.
    #     # y_true = torch.Tensor(y_true)
    #     y_scores = torch.Tensor(y_scores)
    #     error = mean_squared_error(y_true, y_scores,squared=False)
    #     #returning the answer
    #     # return error
    # with torch.no.grad():
        
    #     model.eval()
    #     y_scores = []

    #     for data in train_loader:  # Iterate in batches over the training/test dataset.
    #         out = model(data.x,data.edge_index,data.batch,data.edge_attr).float().flatten()
    #         label = data.y
    #         y_scores.extend(out.numpy())

if __name__=="__main__":
    main()

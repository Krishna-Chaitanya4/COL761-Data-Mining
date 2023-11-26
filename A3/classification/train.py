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

def create_pyg_data(graph_labels,num_nodes,num_edges,node_features,edges,edge_features):
    # Extract graph information for the given index
    dataset = []
    number_of_ones = 0
    number_of_graphs = 0
    nl =0
    el = 0
    for graph_idx in range(len(graph_labels)):
        label = graph_labels.iloc[graph_idx].values[0]
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
        if(np.isnan(label)):
            continue
        if(len(ed)==0):continue
        label = torch.Tensor([label])
        ed = ed.T
        data = Data(edge_attr=edge_fe,x=node_fe,y=label,edge_index=ed)
        number_of_graphs+=1
        number_of_ones+=label
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
    model.train()
    # print('hello')
    tot = 0
    num = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
         optimizer.zero_grad()  # Clear gradients.
         out = model(data.x, data.edge_index, data.batch,data.edge_attr).float()  # Perform a single forward pass.
        #  print(out)
         y = torch.reshape(data.y, (-1, 1))
         loss = criterion(out, y)  # Compute the loss.
        #  print(loss)
         tot +=  loss.item()
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
    model.eval()#evaluater
    # model.eval()
    with torch.no_grad():#nograd
        # predictions = []
        y_true = []#exact array
        y_scores = []#scores
        for data in loader:
            out = model(data.x,data.edge_index,data.batch,data.edge_attr)#model output
            #iojd
            out = torch.sigmoid(out)#apply sigmoid
            label = data.y#label
            out = out.float()#make float
            y_true.extend(label.numpy())#append
            out = out.flatten()#make flat
            # out = ().float().flatten()
            y_scores.extend(out.numpy())#apedn
        y_scores = torch.Tensor(y_scores)#torch
        y_true = torch.Tensor(y_true)#torch
        le = len(y_scores)#length
        su = (y_scores==y_true).sum()#sum
        accuracy = (su/le)#accuracy
        roc_auc = roc_auc_score(y_true,y_scores)#roc_auc loass
        bce_loss = criterion(y_scores, y_true)#bce loss
        return accuracy,bce_loss, roc_auc#return value

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













def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")


    graph_labels = load_csv(os.path.join(args.dataset_path,'graph_labels.csv.gz'))
    num_nodes = load_csv(os.path.join(args.dataset_path,'num_nodes.csv.gz'))
    num_edges = load_csv(os.path.join(args.dataset_path,'num_edges.csv.gz'))
    node_features = load_csv(os.path.join(args.dataset_path,'node_features.csv.gz'))
    edges = load_csv(os.path.join(args.dataset_path,'edges.csv.gz'))
    edge_features = load_csv(os.path.join(args.dataset_path,'edge_features.csv.gz'))
    number_of_graphs,number_of_ones,train_dataset = create_pyg_data(graph_labels,num_nodes,num_edges,node_features,edges,edge_features)

    graph_labels1 = load_csv(os.path.join(args.val_dataset_path,'graph_labels.csv.gz'))
    num_nodes1 = load_csv(os.path.join(args.val_dataset_path,'num_nodes.csv.gz'))
    num_edges1 = load_csv(os.path.join(args.val_dataset_path,'num_edges.csv.gz'))
    node_features1 = load_csv(os.path.join(args.val_dataset_path,'node_features.csv.gz'))
    edges1 = load_csv(os.path.join(args.val_dataset_path,'edges.csv.gz'))
    edge_features1 = load_csv(os.path.join(args.val_dataset_path,'edge_features.csv.gz'))

    number_of_graphs1,number_of_ones1,test_dataset = create_pyg_data(graph_labels1,num_nodes1,num_edges1,node_features1,edges1,edge_features1)
    dataset = CustomDataset(graph_labels,root='./')
    # print(type(dataset))
    # print(dataset.get(0))
    # for i in range(dataset.len()):
        # train_dataset.append(dataset.get(i,graph_labels,num_nodes,num_edges,node_features,edges,edge_features))
    
    Validation_metric_error = []
    Validation_error_in_loss = []
    dataset1 = CustomDataset(graph_labels1,root='./')

    # test_dataset = []
    # for i in range(dataset1.len()):
    #     test_dataset.append(dataset1.get(i,graph_labels1,num_nodes1,num_edges1,node_features1,edges1,edge_features1))

    #Loading the train data
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #Loading the test data
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    Training_metric_error = []
    Training_error_in_loss = []
    
    


    # for step, data in enumerate(train_loader):
    #     print(f'Step {step + 1}:')
    #     print('=======')
    #     print(f'Number of graphs in the current batch: {data.num_graphs}')
    #     print(data)
    #     print()
    train_acc_list = []


    val_acc_list = []

    # model = GATModel(hidden_channels=64)
    # print(model)


    model = GATModel()
    #After creating the model we need to create ythe  loss function
    learning_rate = 0.002#learning rate
    decay = 0.0007#weight decay
    par = model.parameters()#parameters
    optimizer = torch.optim.Adam(par, weight_decay = decay, lr=learning_rate)#optimizer
    # criterion = torch.nn.BCELoss()
    pw = (number_of_graphs - number_of_ones)/number_of_ones
    pw = [pw]
    pw = torch.Tensor(pw)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pw)

    temp = 0
    ans = copy.deepcopy(model)
    n_epoch = 100#epochs
    
    #Now we are learning the data
    for epoch in range(n_epoch):#for loop
        loss = train(model,train_loader,optimizer,criterion)#loss
        #Getting the train values
        train_acc,train_bce,train_roc = test(model,train_loader,criterion)#test on train
        #Getting the test values
        test_acc,test_bce,test_roc = test(model,test_loader,criterion)#test on val
        # Now we are comapring things

        # Training_metric_error.append(train_roc)#adding the values at the end of list
        # Training_error_in_loss.append(train_bce)#adding the values at the end of list
        # #adding the values at the end of list
        # Validation_metric_error.append(test_roc)#adding the values at the end of list
        # #adding the values at the end of list
        # Validation_error_in_loss.append(test_bce)#adding the values at the end of list
        # train_acc_list.append(train_acc)#adding the values at the end of list
        # #adding the values at the end of list
        # val_acc_list.append(test_acc)#Now print the values
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        #Now print test and train values
        print(train_roc,test_roc)
        print(train_bce,test_bce)
        if(test_roc<=temp):continue
        temp = test_roc#better accuracy
        ans = copy.deepcopy(model)

        #Now saving the state
    torch.save({
        'model_state_dict': ans.state_dict(),
        # Other information you may want to save, like optimizer state, training epoch, etc.
    }, args.model_path)

    # value_of_epochs = [i for i in range(n_epoch)]

    # # Plotting ROC-AUC vs Epochs
    # # Set the title of the plot
    # plotting_my_graph_object.title('ROC-AUC vs Epochs')

    # # Set the label for the x-axis
    # plotting_my_graph_object.xlabel('Epochs')

    # # Set the label for the y-axis
    # plotting_my_graph_object.ylabel('ROC-AUC')

    # # Plot the Training data with a label for the legend and red color
    # plotting_my_graph_object.plot(value_of_epochs, Training_metric_error, label='Training', color='red')

    # # Plot the Validation data with a label for the legend and green color
    # plotting_my_graph_object.plot(value_of_epochs, Validation_metric_error, label='Validation Set', color='green')

    # # Display the legend
    # plotting_my_graph_object.legend()

    # # Save the figure with the specified filename
    # plotting_my_graph_object.savefig('roc_auc_classify.png')

    # plotting_my_graph_object.clf()  # Clear the figure

    # # Set the title of the plot
    # plotting_my_graph_object.title('Loss vs Epochs')

    # # Set the label for the x-axis
    # plotting_my_graph_object.xlabel('Epochs')

    # # Set the label for the y-axis
    # plotting_my_graph_object.ylabel('Loss')

    # # Plot the Training data with a label for the legend and red color
    # plotting_my_graph_object.plot(value_of_epochs, Training_error_in_loss, label='Training', color='red')

    # # Plot the Validation data with a label for the legend and green color
    # plotting_my_graph_object.plot(value_of_epochs, Validation_error_in_loss, label='Validation Set', color='green')

    # # Display the legend
    # plotting_my_graph_object.legend()

    # # Save the figure with the specified filename
    # plotting_my_graph_object.savefig('loss_classify.png')

    # plotting_my_graph_object.clf()  # Clear the figure

    # # Set the title of the plot
    # plotting_my_graph_object.title('Accuracy vs Epochs')

    # # Set the label for the x-axis
    # plotting_my_graph_object.xlabel('Epochs')

    # # Set the label for the y-axis
    # plotting_my_graph_object.ylabel('Accuracy')

    # # Plot the Training data with a label for the legend and red color
    # plotting_my_graph_object.plot(value_of_epochs, train_acc_list, label='Training', color='red')

    # # Plot the Validation data with a label for the legend and green color
    # plotting_my_graph_object.plot(value_of_epochs, val_acc_list, label='Validation Set', color='green')

    # # Display the legend
    # plotting_my_graph_object.legend()

    # # Save the figure with the specified filename
    # plotting_my_graph_object.savefig('Accuracy_classify.png')

    # plotting_my_graph_object.clf()  # Clear the figure





if __name__=="__main__":
    main()

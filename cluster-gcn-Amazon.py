from copy import copy
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.data import Data 
from torch_geometric.loader import ClusterData,ClusterLoader
from torch_geometric.nn import GCNConv
from AmazonProducts import AmazonProducts
import os

from logger import Logger

parser = argparse.ArgumentParser(description='GCN')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--best_val_acc', type=float, default=0)
args = parser.parse_args()
print(args)

# Load your homogeneous graph data here
# data = ...
dataset = AmazonProducts(root='/root/icdm/AmazonProducts')
data = dataset[0]
split_idx = dataset.get_idx_split()
#evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)


print(data)

# Convert to Data object for PyG
data = Data(x=data.x, edge_index=data.edge_index, y=data.y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)

print(data)
save_dir = 'processed/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cluster_data = ClusterData(data, num_parts=5000, recursive=True,
                           save_dir='processed/')
train_loader = ClusterLoader(cluster_data, batch_size=500, shuffle=True,
                             num_workers=4)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)

    def inference(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = GCN(data.num_features, args.hidden_channels, dataset.num_classes, 
            args.num_layers, args.dropout).to(device)

#data = data.to(device)

def train(epoch):
    model.train()

    pbar = tqdm(total=data.num_nodes)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask].argmax(dim=-1))
        loss.backward()
        optimizer.step()

        num_examples = batch.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        pbar.update(batch.num_nodes)

    pbar.close()

    return total_loss / total_examples

class MyEvaluator:
    def eval(self, data):
        y_true = data['y_true']
        y_pred = data['y_pred']
        correct = y_pred.eq(y_true).sum().item()
        acc = correct / len(y_true)
        return {'acc': acc}
# Test function remains the same
@torch.no_grad()
def test():
    model.eval()

    pbar = tqdm(total=data.num_nodes)
    pbar.set_description('Test')

    total_examples = 0
    total_correct = 0
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=-1)
        correct = pred.eq(batch.y.argmax(dim=-1)).sum().item()
        total_correct += correct
        total_examples += batch.num_nodes
        pbar.update(batch.num_nodes)

    pbar.close()

    
    # 打印 x_test 和 edge_index_test 的最大值和最小值

    # train_acc = MyEvaluator().eval({
    #     'y_true': data.y[data.train_mask].argmax(dim=-1),
    #     'y_pred': y_pred[data.train_mask].argmax(dim=-1),
    # })['acc']
    # valid_acc = MyEvaluator().eval({
    #     'y_true': data.y[data.val_mask].argmax(dim=-1),
    #     'y_pred': y_pred[data.val_mask].argmax(dim=-1),
    # })['acc']
    test_acc = total_correct / total_examples
    
    return test_acc,test_acc,test_acc


# test_mask = data.test_mask.to(device)
#test()  # Test if inference on GPU succeeds.
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch)
        torch.cuda.empty_cache()
        
        result = test()
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        print(f'Run: {run + 1:02d}, '
              f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
            #   f'Train: {100 * train_acc:.2f}%, '
            #   f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%')
        if valid_acc > args.best_val_acc:
            torch.save(model.state_dict(), 'best_model.pth')
            args.best_val_acc = valid_acc
    logger.print_statistics(run)
logger.print_statistics()





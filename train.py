import argparse

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.metric import Accuracy
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay

from data import ModelNetDataset
from model import CrossEntropyMatrixRegularization, PointNetClassifier


def parse_args():
    parser = argparse.ArgumentParser("Train")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size in training"
    )
    parser.add_argument("--num_category", type=int, default=40, help="ModelNet10/40")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate in training"
    )
    parser.add_argument("--num_point", type=int, default=1024, help="point number")
    parser.add_argument("--max_epochs", type=int, default=200, help="max epochs")
    parser.add_argument("--num_workers", type=int, default=32, help="num wrokers")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--log_batch_num", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="pointnet.pdparams")
    parser.add_argument("--lr_decay_step", type=int, default=20)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.7)
    parser.add_argument(
        "--data_dir", type=str, default="/mnt/data/sata/ssd/datasets/modelnet40_normal_resampled",
    )
    
    parser.add_argument("--bnn", action='store_true')

    return parser.parse_args()


def train(args):
    train_data = ModelNetDataset(args.data_dir, split="train", num_point=args.num_point)
    test_data = ModelNetDataset(args.data_dir, split="test", num_point=args.num_point)
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    model = PointNetClassifier()
    if args.bnn:
        import basic
        model_state_dict = paddle.load(path='pointnet_0.8930.pdparams')
        model.set_state_dict(model_state_dict)
        fp_layers = [id(model.feat.input_transfrom.conv1), id(model.feat.conv1), id(model.fc3)]
        print(id(model))
        model = basic._to_bi_function(model, fp_layers=fp_layers)
        print(id(model))
        print(model)

    scheduler = CosineAnnealingDecay(
        learning_rate=args.learning_rate,
        T_max=args.max_epochs,
    )

    optimizer = Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
    )
    loss_fn = CrossEntropyMatrixRegularization()
    metrics = Accuracy()

    best_test_acc = 0
    for epoch in range(args.max_epochs):        
        metrics.reset()
        model.train()
        for batch_id, data in enumerate(train_loader):
            optimizer.clear_grad()
            x, y = data
            pred, trans_input, trans_feat = model(x)

            loss = loss_fn(pred, y, trans_feat)

            correct = metrics.compute(pred, y)
            metrics.update(correct)
            loss.backward()

            if (batch_id + 1) % args.log_batch_num == 0:
                print(
                    "Epoch: {}, Batch ID: {}, Loss: {}, ACC: {}".format(
                        epoch, batch_id + 1, loss.item(), metrics.accumulate()
                    )
                )
            optimizer.step()
            
        scheduler.step()

        metrics.reset()
        model.eval()
        for batch_id, data in enumerate(test_loader):
            x, y = data
            pred, trans_input, trans_feat = model(x)

            correct = metrics.compute(pred, y)
            metrics.update(correct)
        test_acc = metrics.accumulate()
        print("Test epoch: {}, acc is: {}".format(epoch, test_acc))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            paddle.save(model.state_dict(), args.model_path)
            print("Model saved. Best Test ACC: {}".format(test_acc))
        else:
            print("Model not saved. Current Best Test ACC: {}".format(best_test_acc))


if __name__ == "__main__":
    args = parse_args()
    if args.bnn:
        args.model_path = 'bi' + args.model_path
    print(args)
    train(args)

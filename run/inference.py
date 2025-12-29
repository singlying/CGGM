import torch
import argparse
import numpy as np
from src.model_utils import load_model
from models.msamodel import MSAModel, ClassifierGuided
from src.eval_metrics import eval_iemocap, eval_senti
from datasets.dataloader import getdataloader


def inference():
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to the trained model')
    parser.add_argument('--dataset', type=str, choices=['iemo', 'mosi'], required=True,
                        help='dataset type (iemo or mosi)')
    parser.add_argument('--data_path', type=str, default='',
                        help='path to the test data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    
    args = parser.parse_args()
    
    # 设置设备
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if use_cuda else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    dataloder, orig_dim = getdataloader(args.dataset, args.batch_size, args.data_path)
    test_loader = dataloder['test']
    
    # 加载模型
    model, classifier, hyp_params = load_model(
        args.model_path, 
        MSAModel, 
        ClassifierGuided if hyp_params.modulation != 'none' else None,
        device
    )
    
    # 设置模型为评估模式
    model.eval()
    if classifier is not None:
        classifier.eval()
    
    # 进行推理
    total_loss = 0.0
    results = []
    truths = []
    
    criterion = torch.nn.CrossEntropyLoss() if args.dataset == 'iemo' else torch.nn.L1Loss()
    
    with torch.no_grad():
        for i_batch, batch in enumerate(test_loader):
            text, audio, vision, batch_Y = batch['text'], batch['audio'], batch['vision'], batch['labels']
            eval_attr = batch_Y.squeeze(dim=-1)
            
            if use_cuda:
                text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                if args.dataset == 'iemo':
                    eval_attr = eval_attr.long()
            
            # 前向传播
            preds, _ = model([text, audio, vision])
            
            if args.dataset == 'iemo':
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)
            
            total_loss += criterion(preds, eval_attr).item()
            results.append(preds)
            truths.append(eval_attr)
    
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    
    # 计算评估指标
    results = torch.cat(results)
    truths = torch.cat(truths)
    
    if args.dataset == 'iemo':
        acc, f1 = eval_iemocap(results, truths)
        print(f"测试损失: {avg_loss:.4f}")
        print(f"测试准确率: {acc:.4f}")
        print(f"测试F1分数: {f1:.4f}")
    else:
        mae = eval_senti(results, truths)
        print(f"测试损失: {avg_loss:.4f}")
        print(f"测试MAE: {mae:.4f}")
    
    return results, truths


if __name__ == '__main__':
    inference()
import torch
import os


def load_model(model_path, model_class, classifier_class=None, device='cuda'):
    """
    加载保存的模型
    
    Args:
        model_path: 模型文件路径
        model_class: 模型类
        classifier_class: 分类器类（如果有的话）
        device: 设备类型
    
    Returns:
        model: 加载的模型
        classifier: 加载的分类器（如果有的话）
        hyp_params: 超参数
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    hyp_params = checkpoint['hyp_params']
    
    # 重建模型
    model = model_class(hyp_params.output_dim, hyp_params.orig_dim, hyp_params.proj_dim,
                       hyp_params.num_heads, hyp_params.layers, hyp_params.relu_dropout,
                       hyp_params.embed_dropout, hyp_params.res_dropout, hyp_params.out_dropout,
                       hyp_params.attn_dropout)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    classifier = None
    if classifier_class is not None and 'classifier_state_dict' in checkpoint:
        classifier = classifier_class(hyp_params.output_dim, hyp_params.num_mod, hyp_params.proj_dim, 
                                   hyp_params.num_heads, hyp_params.cls_layers, hyp_params.relu_dropout, 
                                   hyp_params.embed_dropout, hyp_params.res_dropout, hyp_params.attn_dropout)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        classifier = classifier.to(device)
    
    print(f"模型已成功加载: {model_path}")
    print(f"最佳准确率: {checkpoint.get('best_acc', 'N/A')}")
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    
    return model, classifier, hyp_params


def save_model(model, classifier, optimizer, cls_optimizer, epoch, acc, hyp_params, save_path):
    """
    保存模型
    
    Args:
        model: 主模型
        classifier: 分类器（可选）
        optimizer: 优化器
        cls_optimizer: 分类器优化器（可选）
        epoch: 当前轮数
        acc: 准确率
        hyp_params: 超参数
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': acc,
        'hyp_params': hyp_params
    }
    
    if classifier is not None:
        save_dict['classifier_state_dict'] = classifier.state_dict()
        save_dict['cls_optimizer_state_dict'] = cls_optimizer.state_dict()
    
    torch.save(save_dict, save_path)
    print(f"模型已保存到: {save_path}")
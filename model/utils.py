#实现模块的延迟导入
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

#计算模型可供训练的总的参数
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)#numel()函数：返回数组中元素的个数
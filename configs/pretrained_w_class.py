import torch
#加载官方提供的权重文件
pretrained_weights = torch.load('./configs/DDM-DETR_swin_tiny_two_stage_36.pth')
#修改相关权重
num_class = 8#自己数据集分类数
pretrained_weights['model']['transformer.decoder.class_embed.0.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['transformer.decoder.class_embed.0.bias'].resize_(num_class+1)
pretrained_weights['model']['transformer.decoder.class_embed.1.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['transformer.decoder.class_embed.1.bias'].resize_(num_class+1)
pretrained_weights['model']['transformer.decoder.class_embed.2.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['transformer.decoder.class_embed.2.bias'].resize_(num_class+1)
pretrained_weights['model']['transformer.decoder.class_embed.3.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['transformer.decoder.class_embed.3.bias'].resize_(num_class+1)
pretrained_weights['model']['transformer.decoder.class_embed.4.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['transformer.decoder.class_embed.4.bias'].resize_(num_class+1)
pretrained_weights['model']['transformer.decoder.class_embed.5.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['transformer.decoder.class_embed.5.bias'].resize_(num_class+1)
pretrained_weights['model']['transformer.decoder.class_embed.6.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['transformer.decoder.class_embed.6.bias'].resize_(num_class+1)

pretrained_weights['model']['class_embed.0.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.0.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.1.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.1.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.2.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.2.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.3.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.3.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.4.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.4.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.5.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.5.bias'].resize_(num_class+1)
pretrained_weights['model']['class_embed.6.weight'].resize_(num_class+1, 256)
pretrained_weights['model']['class_embed.6.bias'].resize_(num_class+1)
 
# pretrained_weights['model']['query_embed.weight'].resize_(300,512)# 此处50对应生成queries的数量，根据main.py中--num_queries数量修改
torch.save(pretrained_weights, 'de_detr-swin_vedai_%d.pth'%num_class)
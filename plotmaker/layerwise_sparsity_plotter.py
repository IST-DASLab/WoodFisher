import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

cmap = plt.cm.Spectral
mpl.style.use('seaborn')

layerwise_sparsities = \
[0.5208, 0.7101, 0.6547, 0.5857, 0.6438, 0.6693, 0.6857, 0.6358, 0.5888, 0.7375, 0.6032, 0.7315, 0.7498, 0.8114, 0.8562, 0.7949, 0.8306, 0.7495, 0.7625, 0.7114, 0.7141, 0.7232, 0.7499, 0.6459, 0.8260, 0.7364, 0.8620, 0.8395, 0.8445, 0.7684, 0.8499, 0.8436, 0.7894, 0.8119, 0.8386, 0.7944, 0.7917, 0.8345, 0.7886, 0.7534, 0.8218, 0.7494, 0.6707, 0.8735, 0.8010, 0.9251, 0.7945, 0.8581, 0.7891, 0.7065, 0.8803, 0.8450, 0.4825]

layer_names = ['layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv3', 'layer1.0.downsample.0',  'layer1.1.conv1', 'layer1.1.conv2', 'layer1.1.conv3', 'layer1.2.conv1', 'layer1.2.conv2', 'layer1.2.conv3', 'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.conv3', 'layer2.0.downsample.0',  'layer2.1.conv1', 'layer2.1.conv2', 'layer2.1.conv3', 'layer2.2.conv1', 'layer2.2.conv2', 'layer2.2.conv3', 'layer2.3.conv1', 'layer2.3.conv2', 'layer2.3.conv3', 'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.conv3', 'layer3.0.downsample.0',  'layer3.1.conv1', 'layer3.1.conv2', 'layer3.1.conv3', 'layer3.2.conv1', 'layer3.2.conv2', 'layer3.2.conv3', 'layer3.3.conv1', 'layer3.3.conv2', 'layer3.3.conv3', 'layer3.4.conv1', 'layer3.4.conv2', 'layer3.4.conv3', 'layer3.5.conv1', 'layer3.5.conv2', 'layer3.5.conv3', 'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.conv3', 'layer4.0.downsample.0',  'layer4.1.conv1', 'layer4.1.conv2', 'layer4.1.conv3', 'layer4.2.conv1', 'layer4.2.conv2', 'layer4.2.conv3', 'fc']

y_pos = range(len(layerwise_sparsities))
plt.bar(y_pos, layerwise_sparsities)
plt.xticks(y_pos, layer_names, rotation=90)
plt.title('Layerwise sparsity distribution for ResNet-50 on ImageNet @ Overall Sparsity 80%')
plt.tight_layout()
plt.savefig('./sparsity_distribution_plots/resnet_50_imagenet_gradual_80.pdf', format='pdf', dpi=1000, quality=95)
# Rotation of the bars names
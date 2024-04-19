from torchvision.models import resnet50, alexnet, vgg16, densenet161, convnext_base

# write print(model) into a text file
with open('/home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/model_info.txt', 'w') as f:
    f.write(str(resnet50(weights=None)))
    f.write('\n')
    f.write(str(alexnet(weights=None)))
    f.write('\n')
    f.write(str(vgg16(weights=None)))
    f.write('\n')
    f.write(str(densenet161(weights=None)))
    f.write('\n')
    f.write(str(convnext_base(weights=None)))
    f.write('\n')


  




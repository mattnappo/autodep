import torch
import torchvision

def resnet18():
    model = torchvision.models.resnet18(pretrained=True).cpu()
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("resnet18.pt")

def resnet50():
    model = torchvision.models.resnet50(pretrained=True).cpu()
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("resnet50.pt")

def deeplabv3():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    example = torch.rand(1, 3, 500, 500)
    mod = torch.jit.trace(model, example, strict=False)
    mod.save("deeplab_v3.pt")

def faster_rcnn():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    model.eval()
    example = torch.rand(1, 3, 300, 300)
    mod = torch.jit.script(model, example)
    print(mod)
    mod.save("faster_rcnn.pt")

#resnet18()
#resnet50()
#deeplabv3()
faster_rcnn()

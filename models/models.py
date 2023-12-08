import torch
import torchvision

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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
    #example = torch.ones(1, 3, 500, 500, dtype=torch.uint8)
    #y = model(example)
    #print(y)

    input_image = Image.open("../images/cat.png")
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    plt.imshow(r)
    plt.savefig("x.png")

    mod = torch.jit.trace(model, input_batch, strict=False)
    mod.save("new_deeplab_v3.pt")

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
#faster_rcnn()

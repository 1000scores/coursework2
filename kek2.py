from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms as T
import PIL

with Image.open("data/rtsd/autosave01_02_2012_09_13_33.jpg") as im:
    transform = torchvision.transforms.Compose([
        T.ColorJitter(brightness=.5, hue=.3),
        T.RandomPerspective(distortion_scale=0.2, p=0.25),
        T.RandomEqualize(p=0.25),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_train = torchvision.transforms.Compose(
     [torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    im = transform(im)

    im.show()
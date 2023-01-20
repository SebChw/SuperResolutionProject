
from io import BytesIO
from PIL import Image
from torchvision.models import resnet18
from torchvision import transforms
import torch
from fastapi import FastAPI, UploadFile, File
from ray import serve
import ray
import sys
sys.path.append('.')
from src.edsr import get_edsr

app = FastAPI()
ray.init(address="auto")
serve.start(detached=True)


@serve.deployment
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        self.edsr_params = {
            'rgb_range': 1,
            'scaling_factors': [2],
            'n_resblocks': 16,
            'n_feats': 64,
            'res_scale': 1,
            'augment_train': False,
            'loss': 'L2',
        }
        self.count = 0
        self.model = get_edsr(model_parameters=self.edsr_params)
        self.model = resnet18(pretrained=True).eval()
        self.preprocessor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # remove the alpha channel
            transforms.Lambda(lambda t: t[:3, ...]),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, image_payload_bytes):
        pil_image = Image.open(BytesIO(image_payload_bytes))

        pil_images = [pil_image]  # batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images])

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return {"class_index": int(torch.argmax(output_tensor[0]))}

    def super_resolute(self, image_payload_bytes):
        pil_image = Image.open(BytesIO(image_payload_bytes))

    @app.get("/")
    def get(self):
        return "Welcome to the PyTorch model server."

    @app.post("/classify_image")
    async def classify_image(self, file: UploadFile = File(...)):
        image_bytes = await file.read()
        return self.classify(image_bytes)

    @app.post("/super_resolute")
    async def super_resolute(self, file: UploadFile = File(...)):
        image_bytes = await file.read()
        return self.super_resolute(image_bytes)


ModelServer.deploy()

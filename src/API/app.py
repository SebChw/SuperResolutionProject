import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, Response
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from io import BytesIO
from src.edsr import get_edsr
import cv2
import json

app = FastAPI()
ray.init(address="auto")
serve.start(detached=True)


@serve.deployment
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        self.count = 0
        self.params()
        self.model = resnet18(pretrained=True).eval()
        self.preprocessor = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # remove the alpha channel
            # transforms.Lambda(lambda t: t[:3, ...]),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def params(self):
        self.edsr_params = {
            'rgb_range': 1,
            'scaling_factors': [2],
            'n_resblocks': 16,
            'n_feats': 64,
            'res_scale': 1,
            'augment_train': False,
            'loss': 'L2',
            'perceptual_loss': True
        }

    def classify(self, image_payload_bytes):
        pil_image = Image.open(BytesIO(image_payload_bytes))

        pil_images = [pil_image]  # batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images])

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        return {"class_index": int(torch.argmax(output_tensor[0]))}

    def superR(self, image_payload_bytes):
        self.modelS = get_edsr(model_parameters=self.edsr_params)
        self.modelS = self.modelS.load_from_checkpoint(
            "trained_models/edsr.ckpt", strict=False).eval()
        pil_image = Image.open(BytesIO(image_payload_bytes))

        pil_images = [pil_image]  # batch size is one
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = torch.cat(
            [transform(i).unsqueeze(0) for i in pil_images])
        output_tensor = self.modelS(input_tensor)
        array = output_tensor.permute(0, 2, 3, 1).squeeze().detach().numpy()
        # cv2.imshow("y", array)
        # cv2.waitKey(0)
        # image_bytes: bytes = cv2.imencode(".png", array)[1].tobytes()
        return json.dumps(array.tolist())

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
        
        return self.superR(image_bytes)


ModelServer.deploy()

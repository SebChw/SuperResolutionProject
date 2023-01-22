import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from torchsr.models.edsr import EDSR
import json

app = FastAPI()
ray.init(address="auto")
serve.start(detached=True)


@serve.deployment
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        self.count = 0
        self.define_params()
        self.define_models()

    def define_models(self):
        self.modelEDSR = EDSR(model_parameters=self.edsrx2_params)
        self.modelEDSR.load_state_dict(
            torch.load("trained_models/edsrx2CD.pt"))

    def define_params(self):
        self.edsrx2_params = {
            'rgb_range': 1,
            'scaling_factors': [2],
            'n_resblocks': 16,
            'n_feats': 64,
            'res_scale': 1,
            'augment_train': False,
            'loss': 'L2',
            'perceptual_loss': False
        }
        self.edsrx4_params = {
            'rgb_range': 1,
            'scaling_factors': [4],
            'n_resblocks': 16,
            'n_feats': 64,
            'res_scale': 1,
            'augment_train': False,
            'loss': 'L2',
            'perceptual_loss': False
        }
        self.srcnn_params = {
            'augment_train': False,
            'loss': 'L1',
            'perceptual_loss': False,
        }

    def superR(self, image_payload_bytes):
        pil_image = Image.open(BytesIO(image_payload_bytes))

        pil_images = [pil_image]  # batch size is one
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = torch.cat(
            [transform(i).unsqueeze(0) for i in pil_images])

        output_tensor = self.modelEDSR(input_tensor)

        array = output_tensor.permute(0, 2, 3, 1).squeeze().detach().numpy()
        return json.dumps(array.tolist())

    @app.get("/")
    def get(self):
        return "Welcome to the PyTorch Super Resolution model server."

    @app.post("/super_resolute")
    async def super_resolute(self, file: UploadFile = File(...)):
        image_bytes = await file.read()

        return self.superR(image_bytes)


ModelServer.deploy()

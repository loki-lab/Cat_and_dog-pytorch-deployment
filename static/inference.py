import io
import torch
from torchvision import transforms
from PIL import Image
from static.model import ImageClassifier

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def transform_data(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(io.BytesIO(img))
    return transform(image).unsqueeze(0)


def inference_model(tensor):
    model = ImageClassifier()
    model.load_state_dict(torch.load("./static/best_model.pt"))
    model.eval()
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return str(predicted_idx)

# path = "./test/cat.0.jpg"
# images = transform_data(path)
#
# output = inference_model(images)
# print(output)
#

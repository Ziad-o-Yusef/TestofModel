from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models

app = FastAPI()

# تحميل نموذج استخراج الميزات
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# دالة لتحويل الصورة إلى متجه رقمي
def extract_features(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img = transform(img).unsqueeze(0)  # إضافة بعد الدُفعات
    with torch.no_grad():
        features = model(img)
    
    return features.view(-1).numpy().tolist()  # تحويل الميزات إلى قائمة JSON

# نقطة الإدخال لتحويل الصورة إلى متجه
@app.post("/extract/")
async def extract(image: UploadFile = File(...)):
    image_bytes = await image.read()
    feature_vector = extract_features(image_bytes)
    return {"feature_vector": feature_vector}

# تشغيل API محليًا عند الاختبار
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

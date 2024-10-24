from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms, datasets, models
import io

app = FastAPI()

# Densnet, MobileVit 
transforms_1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# EfficientNet
transforms_2 = transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inception Net
transforms_3 = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to 299x299
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

desnenet_model = torch.load("densenet_1epoch.pth")
efficientnet_model = torch.load("efficient_1epoch.pth")
inception_model = torch.load("inception_1epoch.pth")

def predict_densenet(image: Image.Image):
    
    image = transforms_1(image)
    
    
    image = image.unsqueeze(0)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    desnenet_model.to(device)
    
    
    with torch.no_grad():
        
        output = desnenet_model(image)
    
    
    predicted_class_idx = torch.argmax(output, dim=1).item()
    
    
    return predicted_class_idx 


def predict_efficientNet(image: Image.Image):
    
    image = transforms_2(image)
    
    
    image = image.unsqueeze(0)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    efficientnet_model.to(device)
    
    
    with torch.no_grad():
        
        output = efficientnet_model(image)
    
    
    predicted_class_idx = torch.argmax(output, dim=1).item()
    
    
    return predicted_class_idx

def predict_inception(image: Image.Image):
    
    image = transforms_3(image)
    
    
    image = image.unsqueeze(0)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    inception_model.to(device)
    
    
    with torch.no_grad():
        
        output = inception_model(image)
    
    
    predicted_class_idx = torch.argmax(output, dim=1).item()
    
    
    return predicted_class_idx

def predict_mobilevit(image: Image.Image):
    mobilevit_model = torch.load("mobilevit_1epoch.pth") 

    image = transforms_1(image)
    
    
    image = image.unsqueeze(0)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    mobilevit_model.to(device)
    
    
    with torch.no_grad():
        
        output = mobilevit_model(image)
    
    
    predicted_class_idx = torch.argmax(output, dim=1).item()
    
    
    return predicted_class_idx


@app.post("/densenet_predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        
        predicted_class_idx = predict_densenet(image)

        return JSONResponse(content={"predicted_class_idx": predicted_class_idx})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/efficient_predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        
        predicted_class_idx = predict_efficientNet(image)

        return JSONResponse(content={"predicted_class_idx": predicted_class_idx})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    

@app.post("/inception_predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        
        predicted_class_idx = predict_inception(image)

        return JSONResponse(content={"predicted_class_idx": predicted_class_idx})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# TODO:  Eta alada vabe handle kora lagbe 
@app.post("/mobilevit_predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        
        predicted_class_idx = predict_mobilevit(image)

        return JSONResponse(content={"predicted_class_idx": predicted_class_idx})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


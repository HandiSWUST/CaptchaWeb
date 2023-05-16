import torch
import torchvision
from PIL import ImageEnhance
from fastapi import FastAPI
from pydantic import BaseModel

import util

app = FastAPI()
scan_step = 2
window_size = 16
img_width = 90
img_height = 30
label_data = util.load_labels("./assets/LabelData.json")
model_boundary = torch.load("./model/captcha-bound.pt", map_location=torch.device("cpu"))
model_text = torch.load("./model/captcha-text.pt", map_location=torch.device("cpu"))


class Item(BaseModel):
    data: str


@app.post("/api/v2/extension/cr")
async def captcha_recognize(item: Item):
    captcha_str = ""
    img = util.base64_pil(item.data).convert("L")
    i = 0
    while i <= img_width - window_size:
        image = util.cut_img(img, i, window_size, img_height)
        enh_col = ImageEnhance.Contrast(image)
        contrast = 10
        image = enh_col.enhance(contrast)
        tmp_img = torchvision.transforms.ToTensor()(util.resize_fit(image, 32)).unsqueeze(0)
        pred = model_boundary(tmp_img).argmax(dim=1).numpy()[0]
        if pred == 1:
            word_index = model_text(tmp_img).argmax(dim=1).numpy()[0]
            captcha_str += label_data[word_index]
            i += window_size
        else:
            i += 2
    return captcha_str


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

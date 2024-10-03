from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
# from app.ap.model import load_model, predict_and_draw_boxes
import shutil
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# sys.path.append(str("/home/user/Desktop/yolov9/models"))
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from fastapi.middleware.cors import CORSMiddleware 

@smart_inference_mode()
def run(
        model=None,  # model path or triton URL
        source=None,  # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=None,  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        vid_stride=1,
        hide_labels=False,
        hide_conf=False  # video frame-rate stride
):
    source = str(source)
    
   

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0][1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    label = None if hide_labels else (names[c.item()] if hide_conf else f'{names[c.item()]} {conf:.2f}')

                    annotator.box_label(xyxy, label, color=colors(c, True))
            # Stream results
            im0 = annotator.result()
        return im0

            # Save results (image with detections)
            

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return im0
    
    

params = {
    
    'source':  'data/images',
    
    'imgsz': [640],
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'max_det': 1000,
    
    
    'save_txt': False,
    'save_conf': False,
    'save_crop': False,
    'classes': None,
    'agnostic_nms': False,
    'augment': False,
    'visualize': False,
    
    'project': '/home/user/Desktop/yolov9/runs/detect',
    'name': 'exp',
    'exist_ok': False,
    'line_thickness': 3,
    'vid_stride': 1,
}
weights= "../../weights/last.pt"
device='0'
# Expand image size if necessary
params['imgsz'] *= 2 if len(params['imgsz']) == 1 else 1
params['source']="/home/user/Desktop/yolov9/data/Aerial View Vehicle Detection.v1i.yolov8/images/test/2022teknofest-aerial-top-down-of-a-pedestrian-cross-in-a-main-av-2022-04-21-13-39-45-utc-53_jpg.rf.1e0af28fc4798eb1f1e479cd099f192b.jpg"
# Initialize model
half = True
dnn = True
data='../../data/cocoInfer.yaml'
# print(params['device'])
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

# Call the run function with the unpacked dictionary

# Image.fromarray()

app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8501"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # Load the model once when the API starts
# # model = load_model()
# @app.middleware("http")
# def pr (request,call_next):
#     print(request)
#     return call_next

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # image_bytes = await file.read()
    # image = Image.open(io.BytesIO(image_bytes))
    file_location = f"./uploads/{file.filename}"+".jpeg"  # Set a valid file path
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    params['source']=file_location
    # Perform prediction and draw detection boxes
    # result_image = predict_and_draw_boxes(image, model)
    result_image=cv2.cvtColor(run(model=model, **params), cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(result_image)

    # Save the result image to disk
    result_image.save("result_image.jpg")

    return FileResponse("result_image.jpg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
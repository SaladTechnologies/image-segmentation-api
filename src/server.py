import time

start = time.perf_counter()
import os
import torch
from fastapi import FastAPI, File, UploadFile, Response, Depends
from __version__ import __version__
import requests
import cv2
import io
from model import load_model
from pydantic import BaseModel
from typing import Optional
import numpy as np
import json

host = os.environ.get("HOST", "*")
port = int(os.environ.get("PORT", "7999"))
salad_machine_id = os.environ.get("SALAD_MACHINE_ID", "")
salad_container_group_id = os.environ.get("SALAD_CONTAINER_GROUP_ID", "")

model = load_model()


def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    else:
        return "CUDA is not available"


gpu_name = get_gpu_name()

default_response_headers = {
    "X-Salad-Machine-ID": salad_machine_id,
    "X-Salad-Container-Group-ID": salad_container_group_id,
    "X-GPU-Name": gpu_name,
}


def download_image(url):
    r = requests.get(url)
    nparr = np.frombuffer(r.content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


app = FastAPI()


class SegmentRequest(BaseModel):
    point_coords: Optional[list[list[int]]] = None
    point_labels: Optional[list[int]] = None
    multimask_output: Optional[bool] = True
    box: Optional[list[int]] = None


class SegmentURLRequest(SegmentRequest):
    url: Optional[str] = None


@app.get("/hc")
def health_check():
    return {"status": "ok", "version": __version__}


def predict(image, segment_request):
    predict_kwargs = {}
    if segment_request:
        if segment_request.point_coords:
            predict_kwargs["point_coords"] = np.array(segment_request.point_coords)
        if segment_request.point_labels:
            predict_kwargs["point_labels"] = np.array(segment_request.point_labels)
        if segment_request.multimask_output:
            predict_kwargs["multimask_output"] = segment_request.multimask_output
        if segment_request.box:
            predict_kwargs["box"] = np.array(segment_request.box)
    model.set_image(image)
    masks, scores, logits = model.predict(**predict_kwargs)
    masks = masks.tolist()
    scores = scores.tolist()
    logits = logits.tolist()
    return masks, scores, logits


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...), segment_request: SegmentRequest = None
):
    # Read image file as PIL Image
    start = time.perf_counter()
    file_contents = await file.read()
    nparr = np.frombuffer(file_contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_processing_time = time.perf_counter() - start
    start = time.perf_counter()
    masks, scores, logits = predict(image, segment_request)
    inference_time = time.perf_counter() - start
    response_headers = {
        "X-Inference-Time": f"{inference_time:.4f}",
        "X-Image-Load-Time": f"{image_processing_time:.4f}",
        **default_response_headers,
    }
    return Response(
        content=json.dumps({"masks": masks, "scores": scores, "logits": logits}),
        media_type="application/json",
        headers=response_headers,
    )


@app.get("/segment")
async def segment_image_from_url(queries: SegmentURLRequest = Depends()):
    start = time.perf_counter()
    image = download_image(queries.url)
    image_processing_time = time.perf_counter() - start
    start = time.perf_counter()
    masks, scores, logits = predict(image, queries)
    inference_time = time.perf_counter() - start
    response_headers = {
        "X-Inference-Time": f"{inference_time:.4f}",
        "X-Image-Load-Time": f"{image_processing_time:.4f}",
        **default_response_headers,
    }
    return Response(
        content=json.dumps({"masks": masks, "scores": scores, "logits": logits}),
        media_type="application/json",
        headers=response_headers,
    )


stop = time.perf_counter()
print(f"Server Ready in {stop-start:.4f} seconds", flush=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)

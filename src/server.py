import os
import torch
from fastapi import FastAPI, File, UploadFile, Response
from __version__ import __version__

host = os.environ.get("HOST", "*")
port = int(os.environ.get("PORT", "7999"))
salad_machine_id = os.environ.get("SALAD_MACHINE_ID", "")
salad_container_group_id = os.environ.get("SALAD_CONTAINER_GROUP_ID", "")


def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    else:
        return "CUDA is not available"


gpu_name = get_gpu_name()

app = FastAPI()


@app.get("/hc")
def health_check():
    return {"status": "ok", "version": __version__}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)

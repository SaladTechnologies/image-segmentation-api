# image-segmentation-api
A lightweight inference API for Segment Anything

## Download the model

```shell
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P models/
```

## Example use

```bash
curl -X POST "http://localhost:7999/segment" \
     -H  "accept: application/json" \
     -H  "Content-Type: multipart/form-data" \
     -F "file=@test.jpg;type=image/jpeg" \
     -F "segment_request=@test.json;type=application/json"
```


```bash
curl -X GET "http://localhost:7999/segment?url=https%3A%2F%2Fsalad-benchmark-assets.download%2Fcoco2017%2Ftrain2017%2F000000000143.jpg&multimask_output=true" \
     -H  "accept: application/json"
```
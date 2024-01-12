from segment_anything import SamPredictor, sam_model_registry


def load_model():
    # Load model
    sam = sam_model_registry["vit_l"](checkpoint="models/sam_vit_l_0b3195.pth")
    sam.to("cuda")
    return SamPredictor(sam)

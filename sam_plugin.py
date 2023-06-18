# --------------- SHARED ---------------------------------------------------
import sys
from typing import List, Any
sys.path.append('.')  # Add the current directory to the sys path
sys.path.append('utils')  # Add the utils directory to the sys path

from utils.omni_utils_http import CdnResponse, ImageMeta, create_api_route, plugin_main, init_plugin
from pydantic import BaseModel
app, router = init_plugin()
# ---------------------------------------------------------------------------
plugin_module_name = "Plugins.sam_plugin.sam" 

# --------------- ENDPOINT_SAM_GENERATE_MASKS -----------------------------
ENDPOINT_SAM_GENERATE_MASKS = "/sam/generate_masks"

class SamGenerateMasks_Input(BaseModel):
    images: List[CdnResponse]
    detections: List[Any]
    model_type: str
    minimum_score: float
    output_mask:bool
    output_merged: bool
    output_alpha: bool
    invert_mask: bool

    class Config:
        schema_extra = {
            "title": "SAM: Segment Anything"
        }

class SamGenerateMasks_Response(BaseModel):
    media_array: List[CdnResponse]
    json_array: List[Any]

    class Config:
        schema_extra = {
            "title": "SAM: Segment Anything"}
        
SamGenerateMask_Post = create_api_route(
    app=app,
    router=router,
    context=__name__,
    endpoint=ENDPOINT_SAM_GENERATE_MASKS,
    input_class=SamGenerateMasks_Input,
    response_class=SamGenerateMasks_Response,
    handle_post_function="integration_SamGenerateMask_Post",
    plugin_module_name=plugin_module_name
)

endpoints = [ENDPOINT_SAM_GENERATE_MASKS]

# --------------- SHARED ---------------------------------------------------
plugin_main(app, __name__, __file__)
# --------------------------------------------------------------------------
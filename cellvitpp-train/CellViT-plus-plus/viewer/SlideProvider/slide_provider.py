# -*- coding: utf-8 -*-
# FastAPI Backend to Load Slides from OpenSlide and Provide Tiles for OpenLayers
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# @ Lukas Heine, lukas.heine@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import snappy
import ujson as json
from cachetools import TTLCache
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from pathopatch.wsi_interfaces.wsidicomizer_openslide import (
    DeepZoomGeneratorDicom,
    DicomSlide,
)
from PIL import Image
from pydantic import BaseModel
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

SLIDE_PROVIDER_URL = os.getenv("SLIDE_PROVIDER_URL", "http://localhost:3306")
OPENSLIDE_EXTENSIONS = [
    ".svs",
    ".mrxs",
    ".tiff",
    ".czi",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".svslide",
    ".bif",
    ".tif",
]
UPLOAD_DIR = Path("./data")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
global SLIDE_NAME_DISPLAY


# Models
class RegisterSlide(BaseModel):
    image_id: str
    image_path: str


# Documentation
description = """
Viewing Pathology Images using OpenSlide in OpenLayers - FastAPI Backend
"""

tags_metadata = [
    {
        "name": "upload",
        "description": "Upload a new slide, detection, or/and contour file",
    },
    {
        "name": "register",
        "description": "Register new slides by adding the slide ID and the path to the image file",
    },
    {
        "name": "info",
        "description": "Retrieve information about the slide",
    },
    {
        "name": "tiles",
        "description": "Return the tile as a jpg image",
    },
    {
        "name": "detection_exists",
        "description": "Check if a detection file exists on the server",
    },
    {
        "name": "cell_detections",
        "description": "Get cell detections from the database",
    },
    {
        "name": "contour_exists",
        "description": "Check if a contour file exists on the server",
    },
    {
        "name": "cell_contours",
        "description": "Get cell contours from the database",
    },
]

# FastAPI instance
slide_endpoint = FastAPI(
    title="SlideEndpoint",
    description=description,
    summary="API to get tiles for OpenLayers with OpenSlide",
    openapi_tags=tags_metadata,
    version="0.0.1",
    contact={
        "name": "Fabian Hörst",
        "url": "https://mml.ikim.nrw/",
        "email": "fabian.hoerst@uk-essen.de",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
# prevent cors errors
slide_endpoint.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Cache for storing slides
slide_cache = TTLCache(maxsize=20, ttl=180)
slide_paths = {}


# global functions
def get_slide(image_id: str) -> DeepZoomGenerator:
    """Get a slide from the cache or load it from the disk and store it in the cache

    Args:
        image_id (str): ID of the image to load

    Returns:
        DeepZoomGenerator: Slide object
    """
    if image_id in slide_cache:
        slide = slide_cache[image_id]
        logging.info(f"{slide} retrieved from cache")
    else:
        slide_path = Path(slide_paths[image_id])
        if slide_path.suffix.lower() in OPENSLIDE_EXTENSIONS:
            logging.info(f"Load slide with OpenSlide")
            slide = OpenSlide(slide_paths[image_id])
            slide = DeepZoomGenerator(slide, tile_size=256, overlap=0)
            slide_cache[image_id] = slide
            logging.info(f"{slide} stored in cache")
        else:
            try:
                slide_folder = Path(slide_paths[image_id]).parent
                slide_files = [
                    f for f in slide_folder.iterdir() if f.suffix.lower() in [".dcm"]
                ]
                slide = OpenSlide(slide_files[0])
                slide = DeepZoomGenerator(slide, tile_size=256, overlap=0)
                slide_cache[image_id] = slide
                logging.info(f"{slide} stored in cache")
                logging.info("Loaded slide with OpenSlide")
            except:
                logging.info(f"Load slide with DicomSlide")
                slide = DicomSlide(slide_paths[image_id].parent)
                slide = DeepZoomGeneratorDicom(slide, tile_size=256, overlap=0)
                slide_cache[image_id] = slide
                logging.info(f"{slide} stored in cache")
                logging.info("Loaded slide with DicomSlide")

    return slide


def resize_and_fill(
    image: Image,
    target_size: Tuple[int, int] = (256, 256),
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image:
    """Resize an image to a target size and fill the rest with a color

    Args:
        image (Image): Image to resize/fill
        target_size (Tuple[int, int], optional): Output size. Defaults to (256, 256).
        fill_color (Tuple[int, int, int], optional): Color to fill in RGB. Defaults to (255, 255, 255).

    Returns:
        Image: Resized and filled image
    """
    current_size = image.size

    if (current_size[0] < target_size[0]) or (current_size[1] < target_size[1]):
        new_image = Image.new("RGB", target_size, fill_color)
        new_image.paste(image, (0, 0))
        image = new_image
    return image


@slide_endpoint.post("/upload", tags=["upload"])
async def upload_file(req: Request) -> dict:
    """Upload a file to the server

    Args:
        req: Request containing the data from the frontend
    Returns:
        dict: Response with filename and location (keys are "slide_id" and "slide_path")
    """
    # clean cache if there exists a file
    slide_cache.clear()
    slide_paths.clear()

    # cleanup old files
    for file in [f for f in UPLOAD_DIR.iterdir() if f.is_file()]:
        file.unlink()
    for folder in [f for f in UPLOAD_DIR.iterdir() if f.is_dir()]:
        shutil.rmtree(folder)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    form = await req.form()
    wsi_file = form.getlist("wsi-file")[0]
    detection_file = form.getlist("detection-file")[0]
    contour_file = form.getlist("contours-file")[0]

    logger.info("WSI file: ", wsi_file)
    logger.info("Detection file: ", detection_file)
    logger.info("Contour file: ", contour_file)

    # 1. Save the WSI file
    wsi_filename_suffix = Path(wsi_file.filename).suffix
    wsi_filename = f"displayed_wsi{wsi_filename_suffix}"
    wsi_file_location = UPLOAD_DIR / wsi_filename
    with open(wsi_file_location, "wb") as f:
        while chunk := await wsi_file.read(1024 * 1024):
            f.write(chunk)
    # display the name of the slide
    global SLIDE_NAME_DISPLAY
    SLIDE_NAME_DISPLAY = wsi_file.filename

    # 2. If the WSI file is zipped, unzip it
    if wsi_filename_suffix.lower() == ".zip":
        import zipfile

        displayed_wsi_dir = UPLOAD_DIR / "displayed_wsi"
        displayed_wsi_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wsi_file_location, "r") as zip_ref:
            for member in zip_ref.namelist():
                # Ensure only file names are extracted, not folder structures
                member_path = Path(member)
                if not member_path.is_dir():  # Only extract files
                    destination_path = displayed_wsi_dir / member_path.name
                    with destination_path.open("wb") as dest_file:
                        dest_file.write(zip_ref.read(member))
        # remove the zip file
        wsi_file_location.unlink()
        wsi_filename = member_path.name
        wsi_file_location = displayed_wsi_dir / wsi_filename

    # 3. check for detection file (optional file)
    if detection_file != "undefined":
        detection_filename_suffix = Path(detection_file.filename).suffix

        if detection_filename_suffix.lower() == ".geojson":
            detection_filename = f"detection{detection_filename_suffix}"
            detection_file_location = UPLOAD_DIR / detection_filename
            # Directly save the geojson file
            with open(detection_file_location, "wb") as f:
                while chunk := await detection_file.read(1024 * 1024):
                    f.write(chunk)

        elif detection_filename_suffix.lower() == ".snappy":
            detection_filename = (
                "detection.geojson"  # Decompressed file will be stored as .geojson
            )
            detection_file_location = UPLOAD_DIR / detection_filename

            with open(detection_file_location, "wb") as f:
                # Snappy decompression on-the-fly
                while chunk := await detection_file.read(1024 * 1024):
                    # Decompress the current chunk
                    decompressed_chunk = snappy.uncompress(chunk)
                    # Write the decompressed chunk into the .geojson file
                    f.write(decompressed_chunk)

        # assert detection_filename_suffix.lower() == ".geojson", "Detection file must be a geojson file"
        # detection_filename = f"detection{detection_filename_suffix}"
        # detection_file_location = UPLOAD_DIR / detection_filename
        # with open(detection_file_location, "wb") as f:
        #     while chunk := await detection_file.read(1024 * 1024):
        #         f.write(chunk)

    # 4. check for contour file (optional file)
    if contour_file != "undefined":
        contour_filename_suffix = Path(contour_file.filename).suffix
        assert (
            contour_filename_suffix.lower() == ".geojson"
        ), "Contour file must be a geojson file"
        contour_filename = f"contour{contour_filename_suffix}"
        contour_file_location = UPLOAD_DIR / contour_filename
        with open(contour_file_location, "wb") as f:
            while chunk := await contour_file.read(1024 * 1024):
                f.write(chunk)

    return {"slide_id": wsi_filename, "slide_path": str(wsi_file_location.resolve())}


@slide_endpoint.post("/register", tags=["register"])
def register_slide(slide_data: RegisterSlide) -> dict:
    """Register a slide with the backend

    Args:
        slide_data (RegisterSlide): Data of the slide to register

    Returns:
        dict: Response
    """
    slide_paths[slide_data.image_id] = slide_data.image_path
    logging.info(
        f"Registered slide {slide_data.image_id} with path {slide_data.image_path}"
    )
    return {"Image-ID": slide_data.image_id, "Path": slide_data.image_path}


@slide_endpoint.get("/info/{image_id}", tags=["info"])
def return_object_info(image_id: str) -> dict:
    """Return information about the slide for the frontend

    Args:
        image_id (str): ID of the image to get information for

    Returns:
        dict: Information about the slide
    """
    slide = get_slide(image_id)

    extent = slide.level_dimensions[-1]

    level_tiles = np.array(slide.level_tiles)
    min_layer = np.where((level_tiles[:, 0] > 1) & (level_tiles[:, 1] > 1))[0][0]
    min_zoom = int(slide.level_count - min_layer)

    resolutions = [2**i for i in range(slide.level_count)][::-1]
    json_return = {
        "extent": [0, 0, extent[0], extent[1]],
        "slide_url": f"{SLIDE_PROVIDER_URL}/tiles/{image_id}/{{z}}/{{x}}/{{y}}.jpg",
        "minZoom": min_zoom,
        "maxZoom": slide.level_count
        + 1,  # TODO: check if slide.level_count - 1 is better,
        "startZoom": int(np.min([min_zoom + 3, slide.level_count - 3])),
        "resolutions": resolutions,
        "slide_name": SLIDE_NAME_DISPLAY,
    }
    logging.info(json_return)
    return json_return


@slide_endpoint.get("/tiles/{image_id}/{z}/{x}/{y}.jpg", tags=["tile"])
def get_tile_xyz(image_id: str, z: int, x: int, y: int) -> StreamingResponse:
    """Get a tile from the slide

    Args:
        image_id (str): ID of the image to get the tile from
        z (int): Level of the tile
        x (int): Row of the tile
        y (int): Column of the tile

    Returns:
        StreamingResponse: Tile as a streaming response
    """
    slide = get_slide(image_id)
    try:
        tile = slide.get_tile(z, (x, y))
        if tile.size != (256, 256):
            tile = resize_and_fill(
                tile, target_size=(256, 256), fill_color=(255, 255, 255)
            )
    except ValueError:
        logging.info("Requesting non existent address")
        tile = Image.new("RGB", (256, 256), (255, 255, 255))
    tile = tile.convert("RGB")
    img_byte_array = io.BytesIO()
    tile.save(img_byte_array, format="JPEG")
    img_byte_array.seek(0)
    return StreamingResponse(content=img_byte_array, media_type="image/jpeg")


@slide_endpoint.get("/detection_exists", tags=["detection_exists"])
def detection_exists() -> dict:
    """Check if a detection file exists

    Returns:
        dict: Response
    """
    return {"exists": (UPLOAD_DIR / "detection.geojson").exists()}


@slide_endpoint.get("/cell_detections", tags=["cell_detections"])
def get_cell_detections() -> dict:
    """Get cell detections from the database

    Returns:
        dict: Cell detections
    """
    with open(UPLOAD_DIR / "detection.geojson", "r") as f:
        cell_detections = json.load(f)

    return {"cell_detections": cell_detections}


@slide_endpoint.get("/contour_exists", tags=["contour_exists"])
def contour_exists() -> dict:
    """Check if a contour file exists

    Returns:
        dict: Response
    """
    return {"exists": (UPLOAD_DIR / "contour.geojson").exists()}


@slide_endpoint.get("/cell_contours", tags=["cell_contours"])
def get_cell_contours() -> dict:
    """Get cell contour from the database

    Returns:
        dict: Cell Countours
    """
    with open(UPLOAD_DIR / "contour.geojson", "r") as f:
        cell_contours = json.load(f)

    return {"cell_contours": cell_contours}

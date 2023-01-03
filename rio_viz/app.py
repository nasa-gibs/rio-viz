"""rio_viz app."""

import pathlib
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import attr
import rasterio
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query
from geojson_pydantic.features import Feature
from rio_tiler.io import BaseReader, COGReader, MultiBandReader, MultiBaseReader
from rio_tiler.models import BandStatistics, Info
from server_thread import ServerManager, ServerThread
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.types import ASGIApp
from starlette_cramjam.middleware import CompressionMiddleware

from rio_viz.resources.enums import RasterFormat, VectorTileFormat, VectorTileType

from titiler.core.dependencies import (
    AssetsBidxExprParamsOptional,
    AssetsBidxParams,
    AssetsParams,
    BandsExprParamsOptional,
    BandsParams,
    BidxExprParams,
    ColorMapParams,
    DatasetParams,
    DefaultDependency,
    HistogramParams,
    ImageParams,
    ImageRenderingParams,
    PostProcessParams,
    StatisticsParams,
)
from titiler.core.models.mapbox import TileJSON
from titiler.core.resources.responses import JSONResponse, XMLResponse

try:
    from rio_tiler_mvt import pixels_encoder  # noqa

    has_mvt = True
except ModuleNotFoundError:
    has_mvt = False
    pixels_encoder = None

import os
import json
import rasterio
import httpx
from rasterio.features import bounds as featureBounds
from folium import Map, TileLayer, GeoJson
from geojson_pydantic import Feature, Polygon
from rio_tiler.io import COGReader
from cogeo_mosaic.mosaic import MosaicJSON
from cogeo_mosaic.backends import MosaicBackend
import pystac_client
import requests as r
import socket
from pydantic import BaseModel

from fastapi.responses import FileResponse

from os import getcwd

from datetime import datetime

src_dir = str(pathlib.Path(__file__).parent.joinpath("src"))
template_dir = str(pathlib.Path(__file__).parent.joinpath("templates"))
templates = Jinja2Templates(directory=template_dir)

TileFormat = Union[RasterFormat, VectorTileFormat]

# Get container ip 172.21.0.3
# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# s.connect(("8.8.8.8", 80))
# ContainerIp = s.getsockname()[0]

jsonfile = ""

# Check bounds crossinng date line.
# absvalue = 0
# absfile = ""

def cmr_search(msg):
    print("Start STAC search...")

    west_v = msg.west
    east_v = msg.east
    south_v = msg.south
    north_v = msg.north
    date_v = msg.date
    collection_v = msg.collection
    red_v = msg.red
    green_v = msg.green
    blue_v = msg.blue
    scale_v = msg.scale

    # TiTiler server.

    # Must pub ip inside container, not 0.0.0.0.
    # TitilerIp = "titiler"
    # VizexIp = "vizex"
    # titiler_endpoint = "http://" + TitilerIp + ":8000"
    # data_endpoint = "http://" + VizexIp + ":8080"

    # # STAC endpoint.
    # stac_endpoint = 'https://cmr.earthdata.nasa.gov/stac'
    # stac_info = r.get(stac_endpoint).json()
    # for s in stac_info: print(s)
    # print(f"STAC Version: {stac_info['stac_version']}. {stac_info['description']}")
    # print(f"There are {len(stac_info['links'])} STAC catalogs available in CMR.")

    # Select collection.
    # collection_v = ['HLSL30.v2.0', 'HLSS30.v2.0']
    # collection_v = ['HLSL30.v2.0']

    # Print coordinates.
    # roi = json.loads(spatial.to_json())['features'][0]['geometry']
    # roi = {'type': 'Polygon', 'coordinates': [[[-76.1, 37.1], [-76.1, 37.0], [-76.0, 37.0], [-76.0, 37.1], [-76.1, 37.1]]]}
    # roi = {'type': 'Polygon', 'coordinates': [[[west_v, north_v], [west_v, south_v], [east_v, south_v], [east_v, north_v], [west_v, north_v]]]}
    # print("Select spatial area: " + str(roi))

    # Select temporal.

    # date_range = sys.argv[1]

    # date_range = "2022-05-01T00:00:00Z/2022-05-05T23:59:59Z"

    # date_range = "2022-01-01/2022-09-01"

    # l30, DC 18SUJ, 4 images.
    # date_range = "2022-06-17/2022-06-17"

    # s30, DC 18SUJ, 4 images.
    # date_range = "2022-03-01/2022-03-08"
    # date_range = "2022-03-01T00:00:00Z/2022-03-08T00:00:00Z"

    # World.
    # date_range = date_v  # "2022-07-01T00:00:00Z/2022-07-02T00:00:00Z"
    # print("Select date: " + date_range)

    # Open catalog.
    # catalog = pystac_client.Client.open(f'{stac_endpoint}/LPCLOUD/')
    # products = [c for c in catalog.get_children()]

    # search = catalog.search(
    #     collection=collection_v,
    #     intersects=roi,
    #     datetime=date_range,
    #     # limit=100
    # )

    # tile_number = search.matched()
    # print("Tile number: " + str(tile_number))
    # if tile_number == 0:
    #     return "nodata"

    # Limit is 100, but get 113. May just close.
    # item_collection = search.get_all_items()

    # binary, not JSON.
    # print(item_collection.json())

    # List first 5 tiles..
    # print(list(item_collection)[0:5])

    # See all granules in 1 tile. STAC JSON format.

    # Same as string.
    # print(item_collection[0].to_dict())

    # Filters.
    cloudcover = 25

    # s30 13.
    s30_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'Fmask']

    # l30 10.
    l30_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B09', 'B10', 'B11', 'Fmask']

    BandShow = 1

    TileBreaker = False

    # Mosaic files.
    files = []
    boxes = []
    bands = []

    # Original mosaic band.
    mosaic_band = 'B01'

    # Total images in MosaicJSON input file.
    img_number = 0

    # {"features": [
    #     {"geometry": {"coordinates": [[
    #         [84.989696, 26.117197],
    #         [86.097741, 26.117197],
    #         [86.097741, 27.119521],
    #         [84.989696, 27.119521],
    #         [84.989696, 26.117197]
    #     ]],
    #         "type": "Polygon"
    #     },
    #         "properties": {"path": "s3://lp-prod-protected/HLSS30.020/HLS.S30.T45RUK.2022182T044711.v2.0/HLS.S30.T45RUK.2022182T044711.v2.0.B01.tif"}
    #     },
    # ]
    # }

    def file_box_feature(file: str, box: float) -> Feature:
        return Feature(
            geometry=Polygon.from_bounds(
                box[0], box[1], box[2], box[3] #w, s, e, n.
            ),
            properties={
                "path": file,
            }
            # ,
            # type={"Feature"}
        )

    # global absvalue
    # global absfile

    # CMR API.
    cmr = 'https://cmr.earthdata.nasa.gov/search/granules.stac?collection_concept_id='
    if collection_v == 'HLSL30.v2.0' or collection_v == 'HLSS30.v2.0':
        id = 'C2021957295-LPCLOUD'
    # print(id)
    spatial = '&bounding_box='
    area = west_v + ',' + south_v + ',' + east_v + ',' + north_v
    temporal = '&temporal[]='
    # date_v = '2022-07-01T00:00:00Z,2022-07-02T00:00:00Z'
    pagenumber = '&page_num='
    number = 1
    pagesize = '&page_size='
    size = '2000'
    # scroll = '&scroll=true'

    item_n = 2000
    while item_n == 2000:
        url = ""
        if west_v == "":
            url = cmr + id + temporal + date_v + pagenumber + str(number) + pagesize + size
        else:
            url = cmr + id + spatial + area + temporal + date_v + pagenumber + str(number) + pagesize + size
        # url = cmr + id + spatial + area + temporal + date_v + pagenumber + str(number) + pagesize + size + scroll

        requested_data = r.get(url)
        result = json.loads(requested_data.text)
        item_n = len(result["features"])
        print(item_n)
        for item in range(item_n):
            box = result["features"][item]["bbox"]
            assets = result["features"][item]["assets"]
            for asset in assets.values():
                if "B01.tif" in asset["href"]:
                    file = asset["href"].replace("https://data.lpdaac.earthdatacloud.nasa.gov/", "s3://")
                    files.append(file)
                    boxes.append(box)

                    # tmpvalue = abs(box[0] - box[2])
                    # if tmpvalue > absvalue:
                    #     absvalue = tmpvalue
                    #     absfile = file

                    # print(file)
                    # print(box)
                    img_number += 1
        number = number + 1
        # break

    # STAC API.
    # Tiles.
    # for i in item_collection:
    #     BandNumber = 0
    #     # Filter cloud not used for now.
    #     # if i.properties['eo:cloud_cover'] <= cloudcover:
    #     if i.collection_id == 'HLSL30.v2.0':
    #         bands = l30_bands
    #     else:
    #         bands = s30_bands
    #     # Bands.
    #     for a in i.assets:
    #         if any(b == a for b in bands) and a == mosaic_band:
    #             # if any(b==a for b in bands):
    #             # Print band.
    #             # print(i.assets[a].href)
    #             box = i.bbox
    #             file = i.assets[a].href.replace("https://data.lpdaac.earthdatacloud.nasa.gov/", "s3://")
    #             files.append(file)
    #             boxes.append(box)
    #             print(file)
    #             print(box)
    #             # Donwload band.
    #             # os.system("wget " + i.assets[a].href)
    #             img_number += 1
    #             # Comment out for multiple images in one tile.
    #             BandNumber += 1
    #             # Use how many bands.
    #             if BandNumber == BandShow:
    #                 # TileBreaker = True
    #                 # Break inside.
    #                 break
    #     if TileBreaker:
    #         # Break outside.
    #         break

    if img_number == 0:
        return "nodata"
    print("Tile number: " + str(img_number))

    features = [
        file_box_feature(files[f], boxes[f]).dict(exclude_none=True) for f in range(img_number)
    ]

    boundjson = {'features': features, 'type': 'FeatureCollection'}

    boundfile = open("bound.json", 'w')
    json.dump(boundjson, boundfile)

    # One big mosaic bound.
    bounds = featureBounds(boundjson)
    # print(bounds)
    lat = (bounds[3] - bounds[1]) / 2 + bounds[1]
    lon = (bounds[2] - bounds[0]) / 2 + bounds[0]

    # Select area map.
    # m = Map(
    #     location=((float(south_v) + float(north_v)) / 2, (float(west_v) + float(east_v)) / 2),
    #     zoom_start=8
    # )

    # Get zoom level.
    with COGReader(files[0]) as cog:
        info = cog.info()
        print(info.minzoom)
        print(info.maxzoom)

    # HLS.
    # minzoom = 8
    # maxzoom = 12

    # Create mosaic JSON file. MosaicJSON.from_feature look in feature.properties.path to get dataset path.

    now = datetime.now()

    global jsonfile 
    # Use number. No 0 at beginning.
    jsonfile = now.strftime("%Y%m%d%H%M%S")
    jsonfile = now.strftime("%Y%m%d%H%M%S.json")
    print(jsonfile)

    mosaicdata = MosaicJSON.from_features(features, minzoom=info.minzoom, maxzoom=info.maxzoom)

    # with MosaicBackend("/server/" + jsonfile, mosaic_def=mosaicdata) as mosaic:
    # Write to home.
    with MosaicBackend(jsonfile, mosaic_def=mosaicdata) as mosaic:
        mosaic.write(overwrite=True)

    # print(mosaic.info())

    # stac_item = data_endpoint + "/" + jsonfile
    # stac_item = data_endpoint + "/jsondata"

    # print(stac_item)

    # api_json = httpx.get(
    #     f"{titiler_endpoint}/mosaicjson/tilejson.json",
    #     params=(
    #         ("url", stac_item),
    #         ("assets", red_v),
    #         ("assets", green_v),
    #         ("assets", blue_v),
    #         ("minzoom", info.minzoom),
    #         ("maxzoom", info.maxzoom),
    #         ("rescale", scale_v)
    #     )
    # ).json()
    # return api_json["tiles"][0]

    # responseMsg = "[" + str(lat) + "," + str(lon) + "," + jsonfile + "]"

    responseMsg = {'lat': lat,'lon': lon,'file': jsonfile}
    
    # print(absvalue)
    # print(absfile)
    
    return responseMsg

# end of cmr_search.

class CacheControlMiddleware(BaseHTTPMiddleware):
    """MiddleWare to add CacheControl in response headers."""

    def __init__(self, app: ASGIApp, cachecontrol: str = "no-cache") -> None:
        """Init Middleware."""
        super().__init__(app)
        self.cachecontrol = cachecontrol

    async def dispatch(self, request: Request, call_next):
        """Add cache-control."""
        response = await call_next(request)
        if (
            not response.headers.get("Cache-Control")
            and self.cachecontrol
            and request.method in ["HEAD", "GET"]
            and response.status_code < 500
        ):
            response.headers["Cache-Control"] = self.cachecontrol
        return response


@attr.s
class viz:
    """Creates a very minimal slippy map tile server using fastAPI + Uvicorn."""

    src_path: str = attr.ib()
    reader: Union[
        Type[BaseReader], Type[MultiBandReader], Type[MultiBaseReader]
    ] = attr.ib(default=COGReader)

    app: FastAPI = attr.ib(default=attr.Factory(FastAPI))

    port: int = attr.ib(default=8080)
    host: str = attr.ib(default="0.0.0.0")
    config: Dict = attr.ib(default=dict)

    minzoom: Optional[int] = attr.ib(default=None)
    maxzoom: Optional[int] = attr.ib(default=None)
    bounds: Optional[Tuple[float, float, float, float]] = attr.ib(default=None)

    layers: Optional[List[str]] = attr.ib(default=None)
    nodata: Optional[Union[str, int, float]] = attr.ib(default=None)

    # cog / bands / assets
    reader_type: str = attr.ib(init=False)

    router: Optional[APIRouter] = attr.ib(init=False)

    statistics_dependency: Type[DefaultDependency] = attr.ib(init=False)
    layer_dependency: Type[DefaultDependency] = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Update App."""
        self.router = APIRouter()

        if issubclass(self.reader, (MultiBandReader)):
            self.reader_type = "bands"
        elif issubclass(self.reader, (MultiBaseReader)):
            self.reader_type = "assets"
        else:
            self.reader_type = "cog"

        if self.reader_type == "cog":
            # For simple BaseReader (e.g COGReader) we don't add more dependencies.
            self.info_dependency = DefaultDependency
            self.statistics_dependency = BidxExprParams
            self.layer_dependency = BidxExprParams

        elif self.reader_type == "bands":
            self.info_dependency = BandsParams
            self.statistics_dependency = BandsExprParamsOptional
            self.layer_dependency = BandsExprParamsOptional

        elif self.reader_type == "assets":
            self.info_dependency = AssetsParams
            self.statistics_dependency = AssetsBidxParams
            self.layer_dependency = AssetsBidxExprParamsOptional

        with self.reader(self.src_path) as src_dst:
            self.bounds = (
                self.bounds if self.bounds is not None else src_dst.geographic_bounds
            )
            self.minzoom = self.minzoom if self.minzoom is not None else src_dst.minzoom
            self.maxzoom = self.maxzoom if self.maxzoom is not None else src_dst.maxzoom

        self.register_middleware()
        self.register_routes()
        self.app.include_router(self.router)
        self.app.mount("/static", StaticFiles(directory=src_dir), name="static")

    def register_middleware(self):
        """Register Middleware to the FastAPI app."""
        self.app.add_middleware(
            CompressionMiddleware,
            minimum_size=0,
            exclude_mediatype={
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/jp2",
                "image/webp",
            },
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET"],
            allow_headers=["*"],
        )
        self.app.add_middleware(CacheControlMiddleware)

    def _update_params(self, src_dst, options: Type[DefaultDependency]):
        """Create Reader options."""
        if not getattr(options, "expression", None):
            if self.reader_type == "bands":
                # get default bands from self.layers or reader.bands
                bands = self.layers or getattr(src_dst, "bands", None)
                # check if bands is not in options and overwrite
                if bands and not getattr(options, "bands", None):
                    options.bands = bands

            if self.reader_type == "assets":
                # get default assets from self.layers or reader.assets
                assets = self.layers or getattr(src_dst, "assets", None)
                # check if assets is not in options and overwrite
                if assets and not getattr(options, "assets", None):
                    options.assets = assets

    def register_routes(self):  # noqa
        """Register routes to the FastAPI app."""
        img_media_types = {
            "image/png": {},
            "image/jpeg": {},
            "image/webp": {},
            "image/jp2": {},
            "image/tiff; application=geotiff": {},
            "application/x-binary": {},
        }
        mvt_media_types = {
            "application/x-binary": {},
            "application/x-protobuf": {},
        }

        @self.router.get(
            "/info",
            # for MultiBaseReader the output in `Dict[str, Info]`
            response_model=Dict[str, Info] if self.reader_type == "assets" else Info,
            response_model_exclude={"minzoom", "maxzoom", "center"},
            response_model_exclude_none=True,
            response_class=JSONResponse,
            responses={200: {"description": "Return the info of the COG."}},
            tags=["API"],
        )
        def info(params=Depends(self.info_dependency)):
            """Handle /info requests."""
            with self.reader(self.src_path) as src_dst:
                # Adapt options for each reader type
                self._update_params(src_dst, params)
                return src_dst.info(**params)

        @self.router.get(
            "/statistics",
            # for MultiBaseReader the output in `Dict[str, Dict[str, ImageStatistics]]`
            response_model=Dict[str, Dict[str, BandStatistics]]
            if self.reader_type == "assets"
            else Dict[str, BandStatistics],
            response_model_exclude_none=True,
            response_class=JSONResponse,
            responses={200: {"description": "Return the statistics of the COG."}},
            tags=["API"],
        )
        def statistics(
            layer_params=Depends(self.statistics_dependency),
            image_params: ImageParams = Depends(),
            dataset_params: DatasetParams = Depends(),
            stats_params: StatisticsParams = Depends(),
            histogram_params: HistogramParams = Depends(),
        ):
            """Handle /stats requests."""
            with self.reader(self.src_path) as src_dst:
                if self.nodata is not None and dataset_params.nodata is not None:
                    dataset_params.nodata = self.nodata

                # Adapt options for each reader type
                self._update_params(src_dst, layer_params)

                return src_dst.statistics(
                    **layer_params,
                    **dataset_params,
                    **image_params,
                    **stats_params,
                    hist_options={**histogram_params},
                )

        @self.router.get(
            "/point",
            responses={200: {"description": "Return a point value."}},
            response_class=JSONResponse,
            tags=["API"],
        )
        def point(
            coordinates: str = Query(
                ..., description="Coma (',') delimited lon,lat coordinates"
            ),
            layer_params=Depends(self.layer_dependency),
            dataset_params: DatasetParams = Depends(),
        ):
            """Handle /point requests."""
            lon, lat = list(map(float, coordinates.split(",")))
            with self.reader(self.src_path) as src_dst:  # type: ignore
                if self.nodata is not None and dataset_params.nodata is not None:
                    dataset_params.nodata = self.nodata

                # Adapt options for each reader type
                self._update_params(src_dst, layer_params)

                results = src_dst.point(
                    lon,
                    lat,
                    **layer_params,
                    **dataset_params,
                )

            return {"coordinates": [lon, lat], "values": results}

        preview_params = dict(
            responses={
                200: {"content": img_media_types, "description": "Return a preview."}
            },
            response_class=Response,
            description="Return a preview.",
        )

        @self.router.get(r"/preview", **preview_params, tags=["API"])
        @self.router.get(r"/preview.{format}", **preview_params, tags=["API"])
        def preview(
            format: Optional[RasterFormat] = None,
            layer_params=Depends(self.layer_dependency),
            img_params: ImageParams = Depends(),
            dataset_params: DatasetParams = Depends(),
            render_params: ImageRenderingParams = Depends(),
            postprocess_params: PostProcessParams = Depends(),
            colormap: ColorMapParams = Depends(),
        ):
            """Handle /preview requests."""
            with self.reader(self.src_path) as src_dst:  # type: ignore
                if self.nodata is not None and dataset_params.nodata is not None:
                    dataset_params.nodata = self.nodata

                # Adapt options for each reader type
                self._update_params(src_dst, layer_params)

                data = src_dst.preview(
                    **layer_params,
                    **dataset_params,
                    **img_params,
                )
                dst_colormap = getattr(src_dst, "colormap", None)

            if not format:
                format = RasterFormat.jpeg if data.mask.all() else RasterFormat.png

            image = data.post_process(**postprocess_params)

            content = image.render(
                img_format=format.driver,
                colormap=colormap or dst_colormap,
                **format.profile,
                **render_params,
            )

            return Response(content, media_type=format.mediatype)

        part_params = dict(
            responses={
                200: {
                    "content": img_media_types,
                    "description": "Return a part of a dataset.",
                }
            },
            response_class=Response,
            description="Return a part of a dataset.",
        )

        @self.router.get(
            r"/crop/{minx},{miny},{maxx},{maxy}.{format}",
            **part_params,
            tags=["API"],
        )
        @self.router.get(
            r"/crop/{minx},{miny},{maxx},{maxy}/{width}x{height}.{format}",
            **part_params,
            tags=["API"],
        )
        def part(
            minx: float = Path(..., description="Bounding box min X"),
            miny: float = Path(..., description="Bounding box min Y"),
            maxx: float = Path(..., description="Bounding box max X"),
            maxy: float = Path(..., description="Bounding box max Y"),
            format: RasterFormat = Query(
                RasterFormat.png, description="Output image type."
            ),
            layer_params=Depends(self.layer_dependency),
            img_params: ImageParams = Depends(),
            dataset_params: DatasetParams = Depends(),
            render_params: ImageRenderingParams = Depends(),
            postprocess_params: PostProcessParams = Depends(),
            colormap: ColorMapParams = Depends(),
        ):
            """Create image from part of a dataset."""
            with self.reader(self.src_path) as src_dst:  # type: ignore
                if self.nodata is not None and dataset_params.nodata is not None:
                    dataset_params.nodata = self.nodata

                # Adapt options for each reader type
                self._update_params(src_dst, layer_params)

                data = src_dst.part(
                    [minx, miny, maxx, maxy],
                    **layer_params,
                    **dataset_params,
                    **img_params,
                )
                dst_colormap = getattr(src_dst, "colormap", None)

            image = data.post_process(**postprocess_params)

            content = image.render(
                img_format=format.driver,
                colormap=colormap or dst_colormap,
                **format.profile,
                **render_params,
            )

            return Response(content, media_type=format.mediatype)

        feature_params = dict(
            responses={
                200: {
                    "content": img_media_types,
                    "description": "Return part of a dataset defined by a geojson feature.",
                }
            },
            response_class=Response,
            description="Return part of a dataset defined by a geojson feature.",
        )

        @self.router.post(r"/crop", **feature_params, tags=["API"])
        @self.router.post(r"/crop.{format}", **feature_params, tags=["API"])
        @self.router.post(
            r"/crop/{width}x{height}.{format}", **feature_params, tags=["API"]
        )
        def geojson_part(
            geom: Feature,
            format: Optional[RasterFormat] = Query(
                None, description="Output image type."
            ),
            layer_params=Depends(self.layer_dependency),
            img_params: ImageParams = Depends(),
            dataset_params: DatasetParams = Depends(),
            render_params: ImageRenderingParams = Depends(),
            postprocess_params: PostProcessParams = Depends(),
            colormap: ColorMapParams = Depends(),
        ):
            """Handle /feature requests."""
            with self.reader(self.src_path) as src_dst:  # type: ignore
                if self.nodata is not None and dataset_params.nodata is not None:
                    dataset_params.nodata = self.nodata

                # Adapt options for each reader type
                self._update_params(src_dst, layer_params)

                data = src_dst.feature(
                    geom.dict(exclude_none=True), **layer_params, **dataset_params
                )
                dst_colormap = getattr(src_dst, "colormap", None)

            if not format:
                format = RasterFormat.jpeg if data.mask.all() else RasterFormat.png

            image = data.post_process(**postprocess_params)

            content = image.render(
                img_format=format.driver,
                colormap=colormap or dst_colormap,
                **format.profile,
                **render_params,
            )

            return Response(content, media_type=format.mediatype)

        tile_params = dict(
            responses={
                200: {
                    "content": {**img_media_types, **mvt_media_types},
                    "description": "Return a tile.",
                }
            },
            response_class=Response,
            description="Read COG and return a tile",
        )

        @self.router.get(r"/tiles/{z}/{x}/{y}", **tile_params, tags=["API"])
        @self.router.get(r"/tiles/{z}/{x}/{y}.{format}", **tile_params, tags=["API"])
        def tile(
            z: int,
            x: int,
            y: int,
            format: Optional[TileFormat] = None,
            layer_params=Depends(self.layer_dependency),
            dataset_params: DatasetParams = Depends(),
            render_params: ImageRenderingParams = Depends(),
            postprocess_params: PostProcessParams = Depends(),
            colormap: ColorMapParams = Depends(),
            feature_type: Optional[VectorTileType] = Query(
                None,
                title="Feature type (Only for MVT)",
            ),
            tilesize: Optional[int] = Query(None, description="Tile Size."),
        ):
            """Handle /tiles requests."""
            default_tilesize = 256

            if format and format in VectorTileFormat:
                default_tilesize = 128

            tilesize = tilesize or default_tilesize

            with self.reader(self.src_path) as src_dst:  # type: ignore
                if self.nodata is not None and dataset_params.nodata is not None:
                    dataset_params.nodata = self.nodata

                # Adapt options for each reader type
                self._update_params(src_dst, layer_params)

                tile_data = src_dst.tile(
                    x,
                    y,
                    z,
                    tilesize=tilesize,
                    **layer_params,
                    **dataset_params,
                )

                dst_colormap = getattr(src_dst, "colormap", None)

            # Vector Tile
            if format and format in VectorTileFormat:
                if not pixels_encoder:
                    raise HTTPException(
                        status_code=500,
                        detail="rio-tiler-mvt not found, please do pip install rio-viz['mvt']",
                    )

                if not feature_type:
                    raise HTTPException(
                        status_code=500,
                        detail="missing feature_type for vector tile.",
                    )

                content = pixels_encoder(
                    tile_data.data,
                    tile_data.mask,
                    tile_data.band_names,
                    feature_type=feature_type.value,
                )

            # Raster Tile
            else:
                if not format:
                    format = (
                        RasterFormat.jpeg if tile_data.mask.all() else RasterFormat.png
                    )

                image = tile_data.post_process(**postprocess_params)

                content = image.render(
                    img_format=format.driver,
                    colormap=colormap or dst_colormap,
                    **format.profile,
                    **render_params,
                )

            return Response(content, media_type=format.mediatype)

        @self.router.get(
            "/tilejson.json",
            response_model=TileJSON,
            responses={200: {"description": "Return a tilejson"}},
            response_model_exclude_none=True,
            tags=["API"],
        )
        def tilejson(
            request: Request,
            tile_format: Optional[TileFormat] = None,
            layer_params=Depends(self.layer_dependency),  # noqa
            dataset_params: DatasetParams = Depends(),  # noqa
            render_params: ImageRenderingParams = Depends(),  # noqa
            postprocess_params: PostProcessParams = Depends(),  # noqa
            colormap: ColorMapParams = Depends(),  # noqa
            feature_type: str = Query(  # noqa
                None, title="Feature type", regex="^(point)|(polygon)$"
            ),
            tilesize: Optional[int] = Query(None, description="Tile Size."),
        ):
            """Handle /tilejson.json requests."""
            kwargs: Dict[str, Any] = {"z": "{z}", "x": "{x}", "y": "{y}"}
            if tile_format:
                kwargs["format"] = tile_format.value

            tile_url = request.url_for("tile", **kwargs)

            qs = [
                (key, value)
                for (key, value) in request.query_params._list
                if key not in ["tile_format"]
            ]
            if qs:
                tile_url += f"?{urllib.parse.urlencode(qs)}"

            return dict(
                bounds=self.bounds,
                minzoom=self.minzoom,
                maxzoom=self.maxzoom,
                name="rio-viz",
                tilejson="2.1.0",
                tiles=[tile_url],
            )

        @self.router.get(
            "/WMTSCapabilities.xml", response_class=XMLResponse, tags=["API"]
        )
        def wmts(
            request: Request,
            tile_format: RasterFormat = Query(
                RasterFormat.png, description="Output image type. Default is png."
            ),
            layer_params=Depends(self.layer_dependency),  # noqa
            dataset_params: DatasetParams = Depends(),  # noqa
            render_params: ImageRenderingParams = Depends(),  # noqa
            postprocess_params: PostProcessParams = Depends(),  # noqa
            colormap: ColorMapParams = Depends(),  # noqa
            feature_type: str = Query(  # noqa
                None, title="Feature type", regex="^(point)|(polygon)$"
            ),
        ):
            """
            This is a hidden gem.

            rio-viz is meant to be use to visualize your dataset in the browser but
            using this endpoint, you can also load it in you GIS software.

            """
            kwargs = {
                "z": "{TileMatrix}",
                "x": "{TileCol}",
                "y": "{TileRow}",
                "format": tile_format.value,
            }
            tiles_endpoint = request.url_for("tile", **kwargs)

            qs = [
                (key, value)
                for (key, value) in request.query_params._list
                if key not in ["tile_format", "REQUEST", "SERVICE"]
            ]
            if qs:
                tiles_endpoint += f"?{urllib.parse.urlencode(qs)}"

            tileMatrix = []
            for zoom in range(self.minzoom, self.maxzoom + 1):  # type: ignore
                tm = f"""<TileMatrix>
                    <ows:Identifier>{zoom}</ows:Identifier>
                    <ScaleDenominator>{559082264.02872 / 2 ** zoom / 1}</ScaleDenominator>
                    <TopLeftCorner>-20037508.34278925 20037508.34278925</TopLeftCorner>
                    <TileWidth>256</TileWidth>
                    <TileHeight>256</TileHeight>
                    <MatrixWidth>{2 ** zoom}</MatrixWidth>
                    <MatrixHeight>{2 ** zoom}</MatrixHeight>
                </TileMatrix>"""
                tileMatrix.append(tm)

            return templates.TemplateResponse(
                "wmts.xml",
                {
                    "request": request,
                    "tiles_endpoint": tiles_endpoint,
                    "bounds": self.bounds,
                    "tileMatrix": tileMatrix,
                    "title": "Cloud Optimized GeoTIFF",
                    "layer_name": "cogeo",
                    "media_type": tile_format.mediatype,
                },
                media_type="application/xml",
            )

        @self.router.get(
            "/",
            responses={200: {"description": "Simple COG viewer."}},
            response_class=HTMLResponse,
            tags=["Viewer"],
        )
        @self.router.get(
            "/index.html",
            responses={200: {"description": "Simple COG viewer."}},
            response_class=HTMLResponse,
            tags=["Viewer"],
        )
        def viewer(request: Request):
            """Handle /index.html."""
            if self.reader_type == "cog":
                name = "index.html"
            elif self.reader_type == "bands":
                name = "bands.html"
            elif self.reader_type == "assets":
                name = "assets.html"

            return templates.TemplateResponse(
                name=name,
                context={
                    "request": request,
                    "tilejson_endpoint": request.url_for("tilejson"),
                    "stats_endpoint": request.url_for("statistics"),
                    "info_endpoint": request.url_for("info"),
                    "point_endpoint": request.url_for("point"),
                    "allow_3d": has_mvt,
                },
                media_type="text/html",
            )

        @self.router.get(
            "/getmsg",
            tags=["API"],
        )
        def getmsg() -> dict:
            return {
                "res": "res",
                "data": "fromsevernewview",
                "error": "err"
            }

        class Item(BaseModel):
            west: str
            east: str
            south: str
            north: str
            date: str
            collection: str
            red: str
            green: str
            blue: str
            scale: str

        @self.router.post(
            "/search"
        )
        async def search(msg: Item) -> dict:
            print("Get request msg: " + str(msg))
            responseMsg = cmr_search(msg)
            print("Response msg: " + str(responseMsg))

            # import time
            # time.sleep(100)

            return {
                "response": responseMsg
            }

        @self.router.get("/jsondata/{name_file}")
        def getjsondata(name_file: str):
            return FileResponse(path=getcwd() + "/" + name_file)

        @self.router.get(
            "/jsonbound",
            tags=["API"],
        )
        def jsonbound():
            # From home.
            return FileResponse("bound.json")

    @property
    def endpoint(self) -> str:
        """Get endpoint url."""
        return f"http://{self.host}:{self.port}"

    @property
    def template_url(self) -> str:
        """Get simple app template url."""
        return f"http://{self.host}:{self.port}/index.html"

    @property
    def docs_url(self) -> str:
        """Get simple app template url."""
        return f"http://{self.host}:{self.port}/docs"

    def start(self):
        """Start tile server."""
        with rasterio.Env(**self.config):
            uvicorn.run(app=self.app, host=self.host, port=self.port, log_level="info")


@attr.s
class Client(viz):
    """Create a Client usable in Jupyter Notebook."""

    server: ServerThread = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Update App."""
        super().__attrs_post_init__()

        key = f"{self.host}:{self.port}"
        if ServerManager.is_server_live(key):
            ServerManager.shutdown_server(key)

        self.server = ServerThread(self.app, port=self.port, host=self.host)
        ServerManager.add_server(key, self.server)

    def shutdown(self):
        """Stop server"""
        ServerManager.shutdown_server(f"{self.host}:{self.port}")

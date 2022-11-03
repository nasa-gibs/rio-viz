"""rio_viz.cli."""

import importlib
import os
import tempfile
import warnings
from contextlib import ExitStack, contextmanager

import click
import numpy
from rasterio.rio import options
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles
from rio_tiler.io import BaseReader, COGReader, MultiBandReader, MultiBaseReader

from rio_viz import app


@contextmanager
def TemporaryRasterFile(suffix=".tif"):
    """Create temporary file."""
    fileobj = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fileobj.close()
    try:
        yield fileobj
    finally:
        os.remove(fileobj.name)


class NodataParamType(click.ParamType):
    """Nodata index type."""

    name = "nodata"

    def convert(self, value, param, ctx):
        """Validate and parse band index."""
        try:
            if value.lower() == "nan":
                return numpy.nan
            elif value.lower() in ["nil", "none", "nada"]:
                return None
            else:
                return float(value)
        except (TypeError, ValueError):
            raise click.ClickException("{} is not a valid nodata value.".format(value))


@click.command()
@click.argument("src_path", type=str, nargs=1, required=True)
@click.option(
    "--nodata",
    type=NodataParamType(),
    metavar="NUMBER|nan",
    help="Set nodata masking values for input dataset.",
)
@click.option(
    "--minzoom",
    type=int,
    help="Overwrite minzoom",
)
@click.option(
    "--maxzoom",
    type=int,
    help="Overwrite maxzoom",
)
@click.option("--port", type=int, default=8080, help="Webserver port (default: 8080)")
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Webserver host url (default: 0.0.0.0)",
)
@click.option("--no-check", is_flag=True, help="Ignore COG validation")
@click.option(
    "--reader",
    type=str,
    help="rio-tiler Reader (BaseReader). Default is `rio_tiler.io.COGReader`",
)
@click.option(
    "--layers",
    type=str,
    help="limit to specific layers (only used for MultiBand and MultiBase Readers) (e.g --layers b1 --layers b2).",
    multiple=True,
)
@click.option(
    "--server-only",
    is_flag=True,
    default=False,
    help="Launch API without opening the rio-viz web-page.",
)
@click.option(
    "--config",
    "config",
    metavar="NAME=VALUE",
    multiple=True,
    callback=options._cb_key_val,
    help="GDAL configuration options.",
)
def viz(
    src_path,
    nodata,
    minzoom,
    maxzoom,
    port,
    host,
    no_check,
    reader,
    layers,
    server_only,
    config,
):
    """Rasterio Viz cli."""
    if reader:
        module, classname = reader.rsplit(".", 1)
        reader = getattr(importlib.import_module(module), classname)  # noqa
        if not issubclass(reader, (BaseReader, MultiBandReader, MultiBaseReader)):
            warnings.warn(f"Invalid reader type: {type(reader)}")

    dataset_reader = reader or COGReader

    # Check if cog
    with ExitStack() as ctx:
        if (
            src_path.lower().endswith(".tif")
            and not reader
            and not no_check
            and not cog_validate(src_path)[0]
        ):
            # create tmp COG
            click.echo("create temporary COG")
            tmp_path = ctx.enter_context(TemporaryRasterFile())

            output_profile = cog_profiles.get("deflate")
            output_profile.update(dict(blockxsize="256", blockysize="256"))
            config = dict(GDAL_TIFF_INTERNAL_MASK=True, GDAL_TIFF_OVR_BLOCKSIZE="128")
            cog_translate(src_path, tmp_path.name, output_profile, config=config)
            src_path = tmp_path.name

        application = app.viz(
            src_path=src_path,
            reader=dataset_reader,
            port=port,
            host=host,
            config=config,
            minzoom=minzoom,
            maxzoom=maxzoom,
            nodata=nodata,
            layers=layers,
        )
        if not server_only:
            click.echo(f"Viewer started at {application.template_url}", err=True)
            click.launch(application.template_url)

        application.start()

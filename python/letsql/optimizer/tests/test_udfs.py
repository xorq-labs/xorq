import io
import urllib.request
from operator import itemgetter

import pandas as pd
import pyarrow as pa
import pytest
import requests
from PIL import Image


from letsql.internal import SessionContext
from letsql.tests.util import assert_frame_equal

import letsql as ls

IMAGE_FORMAT = "JPEG"
TULIPS_URL = "https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing/blob/master/standard_test_images/tulips.png?raw=true"
SAM_MODEL_URL = "https://storage.googleapis.com/letsql-assets/models/mobile_sam-tiny-vitt.safetensors"


@pytest.fixture
def images_table():
    image = Image.open(requests.get(TULIPS_URL, stream=True).raw)
    output = io.BytesIO()
    image.save(output, format=IMAGE_FORMAT)

    table = pa.Table.from_arrays(
        [
            pa.array(["tulips.png"]),
            pa.array([output.getvalue()], type=pa.binary()),
        ],
        names=["name", "data"],
    )

    return table


def make_images_expr(images_table, path):
    con = ls.connect()
    images = con.register(images_table, table_name="images")
    expr = images.data.segment_anything(str(path), [0.5, 0.6]).name("segmented")
    return expr


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    filename = tmp_path_factory.mktemp("models") / "mobile_sam-tiny-vitt.safetensors"
    filename, _ = urllib.request.urlretrieve(SAM_MODEL_URL, filename)
    return filename


def test_tensor_mean_all():
    context = SessionContext()
    query = "select tensor_mean_all(make_array(1.0, 2.0, 3.0));"
    actual = context.sql(query).to_pandas().rename(lambda x: "tmp", axis="columns")
    expected = pd.DataFrame({"tmp": [2.0]})

    assert_frame_equal(actual, expected)


@pytest.mark.xfail(reason="datafusion 42.0.0 update introduced a bug")
def test_tensor_mean_all_over_matrix():
    context = SessionContext()
    query = "select tensor_mean_all(make_array(make_array(1.0,2.0,3.0), make_array(1.0,2.0,3.0)));"
    actual = context.sql(query).to_pandas().rename(lambda x: "tmp", axis="columns")
    expected = pd.DataFrame({"tmp": [2.0]})

    assert_frame_equal(actual, expected)


def test_segment_anything(images_table, model_path):
    context = SessionContext()
    context.register_record_batches("images", [images_table.to_batches()])
    sql = make_images_expr(images_table, model_path).compile()
    rows = context.sql(sql).to_pylist()
    output = [row for row in map(itemgetter("segmented"), rows)]
    assert all(row["mask"] is not None for row in output)
    assert all(row["iou_score"] is not None for row in output)


def test_segment_anything_op(images_table, model_path):
    sql = make_images_expr(images_table, model_path).compile()
    expected = f"""SELECT SEGMENT_ANYTHING('{model_path}', "t0"."data", MAKE_ARRAY(0.5, 0.6)) AS "segmented" FROM "images" AS "t0\""""
    assert sql == expected


def test_rotate(images_table):
    context = SessionContext()
    context.register_record_batches("images", [images_table.to_batches()])

    rows = context.sql(
        """
        SELECT rotate90(data) as rotated
        FROM images
        LIMIT 3
        """
    ).to_pylist()

    output = [Image.open(io.BytesIO(row["rotated"])) for row in rows]
    assert output is not None
    assert all(image.format == IMAGE_FORMAT for image in output)


def test_rotate_op(images_table):
    con = ls.connect()
    images = con.register(images_table, table_name="images")

    expr = images.data.rotate90()

    rotated = expr.execute()
    assert rotated is not None

    output = [Image.open(io.BytesIO(data)) for data in rotated.to_list()]
    assert output is not None
    assert all(image.format == IMAGE_FORMAT for image in output)

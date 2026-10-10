import math

import pytest

from msp_control.calibration.geometry import Spotsize


def test_circle_area_from_radius():
    area = Spotsize.circle_area(radius=10)

    assert area == pytest.approx(math.pi * 100)


def test_circle_area_from_diameter():
    area = Spotsize.circle_area(diameter=20)

    assert area == pytest.approx(math.pi * 100)


def test_polygon_area():
    # A square with sides of 10 µm has an area of 100 µm².
    area = Spotsize.polygon_area(side_length=10, num_sides=4)

    assert area == pytest.approx(100)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0},
        {"radius": -5},
        {"diameter": 0},
        {"diameter": -10},
    ],
)
def test_circle_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        Spotsize.circle_area(**kwargs)


@pytest.mark.parametrize("num_sides", [2, 13])
def test_polygon_rejects_invalid_number_of_sides(num_sides):
    with pytest.raises(ValueError):
        Spotsize.polygon_area(side_length=10, num_sides=num_sides)


def test_polygon_rejects_invalid_side_length():
    with pytest.raises(ValueError):
        Spotsize.polygon_area(side_length=0, num_sides=6)

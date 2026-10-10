"""Geometric calculations for illuminated areas at the specimen plane."""

import math


class Spotsize:
    @staticmethod
    def circle_area(
        radius: float | None = None,
        diameter: float | None = None,
    ) -> float:
        """Calculate circular area from radius or diameter (µm²)."""
        if radius is not None:
            if radius <= 0:
                raise ValueError("Radius must be positive")
            return math.pi * radius**2

        if diameter is not None:
            if diameter <= 0:
                raise ValueError("Diameter must be positive")
            return math.pi * (diameter / 2) ** 2

        raise ValueError("Either radius or diameter must be provided")

    @staticmethod
    def polygon_area(side_length: float, num_sides: int) -> float:
        """Calculate the area of a regular polygon (µm²)."""
        if not 3 <= num_sides <= 12:
            raise ValueError("Number of sides must be between 3 and 12")

        if side_length <= 0:
            raise ValueError("Side length must be positive")

        return (
            num_sides * side_length**2
            / (4 * math.tan(math.pi / num_sides))
        )

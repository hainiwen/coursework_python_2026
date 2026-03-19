import pytest

from simple_math import poly_first, poly_second, simple_add, simple_div, simple_mult, simple_sub


def test_simple_add():
    assert simple_add(2, 3) == 5
    assert simple_add(-1, 1) == 0


def test_simple_sub():
    assert simple_sub(10, 3) == 7
    assert simple_sub(0, 5) == -5


def test_simple_mult():
    assert simple_mult(4, 5) == 20
    assert simple_mult(-2, 3) == -6


def test_simple_div():
    assert simple_div(10, 2) == 5
    assert simple_div(7, 2) == 3.5


def test_simple_div_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        simple_div(1, 0)


def test_poly_first():
    assert poly_first(2, 1, 3) == 7
    assert poly_first(-1, 4, 2) == 2


def test_poly_second():
    assert poly_second(2, 1, 3, 4) == 23
    assert poly_second(0, 5, 6, 7) == 5

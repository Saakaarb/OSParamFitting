import pytest
import yaml
from tests import helper_functions

def load_tests():
    with open("tests/test_cases.yml", "r") as f:
        return yaml.safe_load(f)["tests"]

def resolve_function(name):
    return getattr(helper_functions, name)

@pytest.mark.parametrize("case", load_tests(), ids=lambda x: x["name"])
def test_functions(case):
    func = resolve_function(case["function"])
    args = case["input"]
    expected = case.get("expected", None)

    # Run the function with the provided arguments
    if args is not None:
        result = func(*args)
    else:
        result = func()

    # If no expected value is provided, just check that no exception was raised
    if expected is None or expected == []:
        print(f"Test '{case['name']}' ran successfully (no expected value to check).")
        assert True
        return

    # Otherwise, compare result to expected (assume numerical)
    import numpy as np
    np_result = np.array(result, dtype="float")
    np_expected = np.array(expected, dtype="float")
    frac_err = np.divide(np_result - np_expected, np_expected)
    standard = np.zeros_like(frac_err)
    assert np.allclose(
        frac_err, standard, rtol=1e-2, atol=1e-1
    ), f"Expected {standard}, got {frac_err}"

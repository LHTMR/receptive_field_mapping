import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from src.components.validation import Validation as Val

class TestValidation(unittest.TestCase):

    def test_validate_path_valid(self):
        Val.validate_path("file.mp4", file_types=[".mp4", ".avi"])

    def test_validate_path_invalid_type(self):
        with self.assertRaises(TypeError):
            Val.validate_path(123, file_types=[".mp4"])

    def test_validate_path_invalid_extension(self):
        with self.assertRaises(ValueError):
            Val.validate_path("file.txt", file_types=[".mp4", ".avi"])

    def test_validate_path_exists_valid(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        try:
            Val.validate_path_exists(tmp.name)
        finally:
            os.remove(tmp.name)

    def test_validate_path_exists_invalid(self):
        with self.assertRaises(FileNotFoundError):
            Val.validate_path_exists("nonexistent_file.txt")

    def test_validate_strings_valid(self):
        Val.validate_strings(a="foo", b="bar")

    def test_validate_strings_invalid(self):
        with self.assertRaises(TypeError):
            Val.validate_strings(a="foo", b=123)

    def test_validate_type_valid(self):
        Val.validate_type(5, int, "TestInt")
        Val.validate_type("abc", str, "TestStr")

    def test_validate_type_invalid(self):
        with self.assertRaises(TypeError):
            Val.validate_type(5, str, "TestStr")

    def test_validate_type_in_list_valid(self):
        Val.validate_type_in_list([1, 2, 3], int, "TestList")

    def test_validate_type_in_list_invalid(self):
        with self.assertRaises(TypeError):
            Val.validate_type_in_list([1, "a"], int, "TestList")

    def test_validate_positive_valid(self):
        Val.validate_positive(5, "TestPos")
        Val.validate_positive(0, "TestPosZero", zero_allowed=True)

    def test_validate_positive_invalid(self):
        with self.assertRaises(ValueError):
            Val.validate_positive(0, "TestPos")
        with self.assertRaises(ValueError):
            Val.validate_positive(-1, "TestPosZero", zero_allowed=True)

    def test_validate_float_in_range_valid(self):
        Val.validate_float_in_range(0.5, 0, 1, "TestFloat")
        Val.validate_float_in_range(1, 0, 1, "TestFloat")

    def test_validate_float_in_range_invalid(self):
        with self.assertRaises(ValueError):
            Val.validate_float_in_range("a", 0, 1, "TestFloat")
        with self.assertRaises(ValueError):
            Val.validate_float_in_range(2, 0, 1, "TestFloat")

    def test_validate_in_list_valid(self):
        Val.validate_in_list("a", ["a", "b"], "TestList")

    def test_validate_in_list_invalid(self):
        with self.assertRaises(ValueError):
            Val.validate_in_list("c", ["a", "b"], "TestList")

    def test_validate_array_valid(self):
        arr = np.zeros((2, 2))
        Val.validate_array(arr, shape=(2, 2), name="TestArray")

    def test_validate_array_invalid(self):
        with self.assertRaises(ValueError):
            Val.validate_array([1, 2], shape=(2, 2), name="TestArray")
        with self.assertRaises(ValueError):
            Val.validate_array(np.zeros((3, 2)), shape=(2, 2), name="TestArray")

    def test_validate_array_int_float_valid(self):
        arr = np.array([[1, 2], [3, 4]], dtype=int)
        Val.validate_array_int_float(arr, shape=(2, 2), name="TestArray")
        arrf = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        Val.validate_array_int_float(arrf, shape=(2, 2), name="TestArray")

    def test_validate_array_int_float_invalid(self):
        arr = np.array([[1, 2], [3, 4]], dtype=object)
        with self.assertRaises(ValueError):
            Val.validate_array_int_float(arr, shape=(2, 2), name="TestArray")
        with self.assertRaises(ValueError):
            Val.validate_array_int_float(np.zeros((3, 2)), shape=(2, 2), name="TestArray")

    def test_validate_list_int_valid(self):
        Val.validate_list_int([1, 2, 3], shape=(3,), name="TestList")

    def test_validate_list_int_invalid(self):
        with self.assertRaises(ValueError):
            Val.validate_list_int([1, "a", 3], shape=(3,), name="TestList")
        with self.assertRaises(ValueError):
            Val.validate_list_int([1, 2], shape=(3,), name="TestList")
        with self.assertRaises(ValueError):
            Val.validate_list_int("notalist", shape=(3,), name="TestList")

    def test_validate_dataframe_valid(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mapping = Val.validate_dataframe(df, required_columns=["A", "B"], name="TestDF")
        self.assertIn("A", mapping)
        self.assertIn("B", mapping)

    def test_validate_dataframe_invalid(self):
        df = pd.DataFrame({"A": [1, 2]})
        with self.assertRaises(ValueError):
            Val.validate_dataframe(df, required_columns=["A", "B"], name="TestDF")

    def test_validate_dataframe_numeric_valid(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0]})
        Val.validate_dataframe_numeric(df, name="TestDF")

    def test_validate_dataframe_numeric_invalid(self):
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        with self.assertRaises(ValueError):
            Val.validate_dataframe_numeric(df, name="TestDF")
        df_empty = pd.DataFrame()
        with self.assertRaises(ValueError):
            Val.validate_dataframe_numeric(df_empty, name="TestDF")

if __name__ == "__main__":
    unittest.main()
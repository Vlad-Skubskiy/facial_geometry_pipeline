import pytest
import numpy as np
from geometry import FacialGeometryPipeline

@pytest.fixture
def measurer():
    return FacialGeometryPipeline()

def test_no_face_detection(measurer):

    black_image = np.zeros((500, 500, 3), dtype=np.uint8)
    result = measurer.process_image(black_image)

    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"] == "No face detected"

def test_invalid_input_shape(measurer):
    invalid_image = np.zeros((100, 100), dtype=np.uint8)
    try:
        result = measurer.process_image(invalid_image)
    except Exception as e:
        pytest.fail(f"Method is not complete instead of {e}")
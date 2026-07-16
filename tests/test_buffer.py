import numpy as np
import pytest
from eeg_simulator.buffer import RingBuffer


def test_buffer_initialization():
    buf = RingBuffer(max_samples=10, num_channels=4)
    assert buf.max_samples == 10
    assert buf.num_channels == 4
    assert buf.total_samples_written == 0


def test_push_shape_guard():
    buf = RingBuffer(max_samples=5, num_channels=8)
    with pytest.raises(ValueError):
        buf.push(np.array([1.0, 2.0, 3.0]))


def test_chronological_window_retrieval():
    buf = RingBuffer(max_samples=5, num_channels=2)
    
    buf.push(np.array([1.0, 1.0]))
    buf.push(np.array([2.0, 2.0]))
    
    window = buf.get_latest_window(window_size=2)
    
    # Using assert_allclose to handle float32 precision comparison nicely
    np.testing.assert_allclose(window[0], [1.0, 1.0], rtol=1e-5)
    np.testing.assert_allclose(window[1], [2.0, 2.0], rtol=1e-5)


def test_buffer_wrap_around():
    buf = RingBuffer(max_samples=3, num_channels=1)
    
    buf.push(np.array([10.0]))
    buf.push(np.array([20.0]))
    buf.push(np.array([30.0]))
    buf.push(np.array([40.0])) 
    
    window = buf.get_latest_window(window_size=3)
    expected = np.array([[20.0], [30.0], [40.0]], dtype=np.float32)
    np.testing.assert_allclose(window, expected, rtol=1e-5)


def test_empty_buffer_retrieval():
    buf = RingBuffer(max_samples=10, num_channels=2)
    window = buf.get_latest_window(window_size=5)
    assert window.shape == (0, 2)
    assert isinstance(window, np.ndarray)


def test_partial_window_before_filled():
    buf = RingBuffer(max_samples=10, num_channels=2)
    buf.push(np.array([1.0, 2.0]))
    
    window = buf.get_latest_window(window_size=5)
    assert window.shape == (1, 2)


def test_invalid_window_requests():
    buf = RingBuffer(max_samples=10, num_channels=2)
    
    with pytest.raises(ValueError, match="positive"):
        buf.get_latest_window(0)
    
    with pytest.raises(ValueError, match="capacity"):
        buf.get_latest_window(11)


def test_invalid_constructor_params():
    with pytest.raises(ValueError):
        RingBuffer(max_samples=0, num_channels=2)
    with pytest.raises(ValueError):
        RingBuffer(max_samples=10, num_channels=0)

def test_binary_bytes_retrieval():
    """Verify raw byte arrays are packed as float32 sequences (4 bytes per element)."""
    buf = RingBuffer(max_samples=5, num_channels=2)
    buf.push(np.array([1.0, 2.0]))
    
    raw_bytes = buf.get_latest_window_bytes(window_size=1)
    
    # 2 channels * 4 bytes per float32 = 8 bytes total
    assert len(raw_bytes) == 8
    
    # Unpack to verify exact values
    floats_decoded = np.frombuffer(raw_bytes, dtype=np.float32)
    np.testing.assert_allclose(floats_decoded, [1.0, 2.0], rtol=1e-5)


def test_binary_bytes_wrap_around():
    """Verify wrapping binary extraction reconstructs linear byte windows correctly."""
    buf = RingBuffer(max_samples=2, num_channels=1)
    buf.push(np.array([1.0]))
    buf.push(np.array([2.0]))
    buf.push(np.array([3.0])) # Overwrites [1.0]
    
    raw_bytes = buf.get_latest_window_bytes(window_size=2)
    
    # 2 samples * 1 channel * 4 bytes = 8 bytes
    assert len(raw_bytes) == 8
    floats_decoded = np.frombuffer(raw_bytes, dtype=np.float32)
    np.testing.assert_allclose(floats_decoded, [2.0, 3.0], rtol=1e-5)
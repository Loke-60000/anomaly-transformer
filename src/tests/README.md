# Tests

This directory contains unit tests for the transformer anomaly detection system.

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
# From project root
pytest tests/

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run specific test class
pytest tests/test_config.py::TestDataConfig

# Run specific test method
pytest tests/test_config.py::TestDataConfig::test_data_config_creation

# Verbose output
pytest tests/ -v

# Show print statements
pytest tests/ -s
```

## Test Structure

```
tests/
├── __init__.py                 # Test package init
├── test_config.py              # Configuration tests
├── test_data.py                # Data loading/preprocessing tests
├── test_models.py              # Model architecture tests
└── README.md                   # This file
```

## Test Coverage

### test_config.py
- **DataConfig**: Validation, defaults, file checking
- **ModelConfig**: Model type selection, validation
- **TrainingConfig**: Device selection, parameter validation
- **InferenceConfig**: Threshold validation, device selection
- **VisualizationConfig**: Output settings validation
- **PipelineConfig**: Full configuration creation and serialization
- **ConfigurationManager**: Save/load functionality
- **PresetConfigurations**: All preset configurations

### test_data.py
- **DataPreprocessor**: Data cleaning, validation, interpolation
- **TimeSeriesDataset**: Window creation, indexing, PyTorch integration
- **JSONDataLoader**: JSON file loading
- **AnomalyDataLoader**: End-to-end data pipeline, normalization, train/val split

### test_models.py
- **PositionalEncoding**: Encoding creation and application
- **TransformerAutoencoder**: Full model forward pass, encoding, reconstruction
- **LightweightTransformerAutoencoder**: Lightweight model, bottleneck compression
- **Model Factory**: Model creation from config
- **Model Comparison**: Parameter counts, output shapes

## Writing New Tests

### Test Naming Convention
- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>`

### Example Test

```python
import pytest

class TestMyClass:
    """Test MyClass functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality works."""
        obj = MyClass()
        result = obj.do_something()
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error handling."""
        obj = MyClass()
        with pytest.raises(ValueError, match="error message"):
            obj.do_invalid_thing()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return [1, 2, 3, 4, 5]
    
    def test_with_fixture(self, sample_data):
        """Test using fixture."""
        obj = MyClass()
        result = obj.process(sample_data)
        assert len(result) == len(sample_data)
```

## Continuous Integration

These tests can be integrated with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src --cov-report=xml
```

## Test Data

Tests use temporary directories and synthetic data to avoid dependencies on external files.

## Coverage Goals

- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test components working together (not yet implemented)
- **Target coverage**: >80% for all modules

## Known Limitations

- Tests currently mock training loops (full training tests would be too slow)
- Inference tests use small models and synthetic data
- GPU tests only run if CUDA is available

## Troubleshooting

**Import errors**: Make sure you're running pytest from the project root directory.

**Fixture not found**: Check that pytest is discovering the conftest.py file.

**Slow tests**: Use pytest-xdist for parallel execution: `pytest tests/ -n auto`

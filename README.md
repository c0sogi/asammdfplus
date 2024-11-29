# asammdfplus

[![PyPI Version](https://img.shields.io/pypi/v/asammdfplus.svg)](https://pypi.org/project/asammdfplus/)
[![License](https://img.shields.io/pypi/l/asammdfplus.svg)](https://github.com/c0sogi/asammdfplus/blob/main/LICENSE)

`asammdfplus` is an extension of the [asammdf](https://github.com/danielhrisca/asammdf) library, providing additional functionalities for handling Measurement Data Format (MDF) files in Python. It offers enhanced methods for data manipulation, plotting, and input/output operations, making it easier to work with large datasets commonly used in automotive and industrial applications.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Loading MDF Files](#loading-mdf-files)
  - [Accessing Signals](#accessing-signals)
  - [Data Manipulation](#data-manipulation)
  - [Plotting Signals](#plotting-signals)
  - [Converting to DataFrame](#converting-to-dataframe)
  - [Cutting and Concatenating Data](#cutting-and-concatenating-data)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Enhanced MDF Class**: `MDFPlus` extends the `asammdf.MDF` class with additional methods and properties.
- **Easy Signal Access**: Access signals directly using dictionary-like syntax.
- **DataFrame Integration**: Convert MDF data to Pandas DataFrames for easy data analysis.
- **Advanced Plotting**: Plot multiple signals with customized styles using Matplotlib.
- **Signal Manipulation**: Methods for cutting, filtering, and resampling signals.
- **Cache Mechanism**: Caching of signals to improve performance when accessing signals multiple times.
- **IO Utilities**: Functions for reading from and writing to MDF and DataFrame formats.

## Installation

You can install `asammdfplus` via pip:

```bash
pip install asammdfplus
```

**Note**: `asammdfplus` requires Python 3.10 or later.

## Usage

### Loading MDF Files

```python
from asammdfplus import MDFPlus

# Load an MDF file
mdf = MDFPlus('data.mf4')
```

### Accessing Signals

#### Using Dictionary Syntax

```python
# Access a signal directly
signal_series = mdf['EngineSpeed']

# Set or overwrite a signal
mdf['NewSignal'] = signal_series * 2
```

#### Checking Signal Existence

```python
if 'VehicleSpeed' in mdf:
    print("Signal exists in the MDF file.")
```

### Data Manipulation

#### Getting Signal Information

```python
# Get a signal as a pandas Series
engine_speed_series = mdf['EngineSpeed']

# Get the start and end times of the MDF file
start_time, end_time = mdf.startend
```

#### Cutting Signals

```python
# Cut the MDF file between 10 and 20 seconds
mdf_cut = mdf.cut(start=10, stop=20)
```

#### Filtering Signals

```python
# Keep only specified signals
mdf_filtered = mdf.filter(channels=['EngineSpeed', 'VehicleSpeed'])
```

### Plotting Signals

```python
from asammdfplus import plot

# Define groups of signals to plot
groups = {
    'Speed Signals': ['EngineSpeed', 'VehicleSpeed'],
    'Temperature Signals': ['CoolantTemp', 'OilTemp'],
}

# Plot the signals
figs = mdf.plot(groups=groups)
```

#### Customizing Plots

```python
# Advanced plotting with customization
figs = mdf.plot(
    groups=groups,
    fig_cols=2,
    figsize_per_row=(12, 4),
    cmap='viridis',
    line_styles={'EngineSpeed': '--', 'VehicleSpeed': '-.'},
    markers={'CoolantTemp': 'o', 'OilTemp': 's'},
    grid=True,
)
```

### Converting to DataFrame

```python
# Convert the entire MDF file to a DataFrame
df = mdf.to_dataframe()

# Convert specific channels with resampling
df_resampled = mdf.to_dataframe(channels=['EngineSpeed'], raster=0.1)
```

### Cutting and Concatenating Data

#### Cutting with Intervals

```python
# Define intervals to cut
intervals = [(10, 20), (30, 40)]

# Cut the MDF file with the intervals
mdf_cut = mdf.cut(timestamps=intervals)
```

#### Concatenating MDF Files

```python
# Concatenate multiple MDF files
mdf_combined = MDFPlus.concatenate(['file1.mf4', 'file2.mf4'])
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Write tests for your changes.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License.

---

For more information, please visit the [GitHub repository](https://github.com/c0sogi/asammdfplus).
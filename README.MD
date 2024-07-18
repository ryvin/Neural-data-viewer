# Neural Data Analysis Program

![NDAP](/images/NDAP.png)

## Overview

This Neural Data Analysis Program is designed to process and analyze neural data from the hc-1 dataset available at the Collaborative Research in Computational
Neuroscience (CRCNS) database. [Data Sets → Hippocampus → hc-1 → About hc-1](https://crcns.org/data-sets/hc/hc-1/about-hc-1). It provides functionality for loading multiple data files, visualizing the data with interactive zooming, detecting spikes, and comparing the results with information from the IntraExtra.xls file.

This program was created by Raul Pineda to read Neural Scan Data binaries, while importing specific data set criteria from project specific XML documents to be used in algorithms development and analysis.

## How to cite the data and conditions for usage of this data in publications
Any publications made using the example dataset provided should cite the following publication:

Henze, DA; Harris, KD; Borhegyi, Z; Csicsvari, J; Mamiya, A; Hirase, H; Sirota, A; Buzsáki, G (2009): Simultaneous intracellular and extracellular recordings from hippocampus region CA1 of anesthetized rats. CRCNS.org.
http://dx.doi.org/10.6080/K02Z13FP

The above citation uses a Digital Object Identifier (DOI) which is assigned to the data set.  The DOI was created using DataCite (www.datacite.org) and the California Digital Library, "EZID" system (n2t.net/ezid/).

## Features

- Automated loading of multiple .dat files for a given cell
- Parsing of XML configuration files for each data file
- Data visualization with interactive zooming capabilities
- Spike detection on intracellular channels
- Comparison of actual data with IntraExtra.xls information
- Flexible channel selection for visualization

## How It Works

1. **Data Loading**: The program reads all .dat files for a specified cell, along with their corresponding .xml configuration files.

2. **Data Processing**: 
   - Converts raw data to voltage based on parameters in the XML files
   - Combines data from multiple files into a single array

3. **Visualization**: 
   - Plots selected channels using matplotlib
   - Implements interactive zooming for detailed analysis

4. **Spike Detection**: 
   - Performs basic threshold-based spike detection on the intracellular channel

5. **Data Comparison**: 
   - Compares analyzed data with information from the IntraExtra.xls file

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- SciPy
- openpyxl (for .xlsx files)
- xlrd (for .xls files)

## Setup

1. Clone this repository or download the script to your local machine.

2. Install the required Python libraries:

   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the following files in your project directory:
   - `neural_data_analysis.py` (the main script)
   - `IntraExtra.xls` (or .xlsx) file from the CRCNS hc-1 dataset

4. Organize your data files:
   - Create a subdirectory for each cell (e.g., `d5331`)
   - Place the `.dat` and corresponding `.xml` files in the cell's directory

## Usage

1. Open a terminal or command prompt and navigate to the directory containing the script.

2. Run the script with Python, specifying the cell name you want to analyze:

   ```
   python neural_data_analysis.py d5331
   ```

   Replace `d5331` with the name of the cell you want to analyze.

3. The script will:
   - Load all .dat files for the specified cell
   - Process and combine the data
   - Display a plot of the selected channels
   - Print spike detection results and data comparisons

4. Interacting with the plot:
   - Click and drag to zoom in on a specific time range
   - Use the matplotlib toolbar to pan, zoom, or reset the view

## Customization

You can customize the analysis by modifying the following parameters in the script:

- `channels_to_plot`: Specify which channels to visualize
- `threshold` in `detect_spikes` function: Adjust the spike detection sensitivity
- `data_directory`: Change if your data is not in the current directory

## Troubleshooting

- If you encounter file not found errors, ensure your data files are in the correct directory structure.
- For memory issues with large datasets, consider processing files in smaller batches.

## Contributing

Feel free to fork this repository and submit pull requests with any improvements or additional features you develop.

## License

This project is open-source and available under the Apache 2.0 License.
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import pandas as pd
from scipy import signal

class DatasetInfo:
    def __init__(self, excel_file="IntraExtra.xls"):
        self.excel_file = excel_file
        self.df = self._read_excel_file()
    
    def _read_excel_file(self):
        _, ext = os.path.splitext(self.excel_file)
        if ext.lower() == '.xls':
            engine = 'xlrd'
        elif ext.lower() == '.xlsx':
            engine = 'openpyxl'
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return pd.read_excel(self.excel_file, sheet_name="Sheet1", engine=engine)
    
    def get_cell_info(self, cell):
        cell = "D" + cell[1:]  # Ensure cell name starts with "D"
        cell_info = self.df[self.df['cell'] == cell]
        if not cell_info.empty:
            return cell_info.iloc[0].to_dict()
        return None

def parse_xml_config(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    acquisition = root.find('acquisitionSystem')
    nChannels = int(acquisition.find('nChannels').text)
    samplingRate = float(acquisition.find('samplingRate').text)
    nBits = int(acquisition.find('nBits').text)
    voltageRange = float(acquisition.find('voltageRange').text)
    amplification = float(acquisition.find('amplification').text)
    offset = int(acquisition.find('offset').text)
    
    channels = []
    for channel in root.findall('.//channels/channelColors'):
        channel_id = int(channel.find('channel').text)
        color = channel.find('color').text
        channels.append({
            'id': channel_id,
            'name': f'Channel {channel_id}',
            'color': color,
            'enabled': True
        })
    
    return {
        'nChannels': nChannels,
        'samplingRate': samplingRate,
        'nBits': nBits,
        'voltageRange': voltageRange,
        'amplification': amplification,
        'offset': offset,
        'channels': channels
    }

def load_data(file_path, config):
    data = np.fromfile(file_path, dtype=np.int16)
    data = data.reshape(-1, config['nChannels'])
    
    # Convert to voltage
    voltage_range = config['voltageRange']
    nBits = config['nBits']
    amplification = config['amplification']
    offset = config['offset']
    
    data = (data - offset) * (voltage_range / (2**nBits)) / amplification * 1000  # Convert to mV
    
    return data

def plot_data_with_zoom(data, config, channels_to_plot=None):
    if channels_to_plot is None:
        channels_to_plot = range(config['nChannels'])
    
    time = np.arange(0, data.shape[0]) / config['samplingRate']
    
    fig, axs = plt.subplots(len(channels_to_plot), 1, figsize=(15, 4 * len(channels_to_plot)), sharex=True)
    if len(channels_to_plot) == 1:
        axs = [axs]
    
    for i, channel in enumerate(channels_to_plot):
        axs[i].plot(time, data[:, channel], color=config['channels'][channel]['color'])
        axs[i].set_ylabel(f'Channel {channel} (mV)')
        axs[i].set_title(f'Channel {channel}')
    
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    def on_zoom(event):
        ax = event.inaxes
        if ax is None:
            return
        xlim = ax.get_xlim()
        for axis in axs:
            axis.set_xlim(xlim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_release_event', on_zoom)
    plt.show()

def detect_spikes(data, channel, threshold=3.5, ref_period=0.001, config=None):
    if config is None:
        raise ValueError("Config must be provided for spike detection")
    
    signal = data[:, channel]
    mean = np.mean(signal)
    std = np.std(signal)
    threshold_value = mean + threshold * std
    
    above_threshold = np.where(signal > threshold_value)[0]
    spike_times = []
    last_spike = -1
    
    for i in above_threshold:
        if i - last_spike > ref_period * config['samplingRate']:
            spike_times.append(i)
            last_spike = i
    
    return np.array(spike_times) / config['samplingRate']

def analyze_data(cell_name, data_directory, dataset_info, channels_to_plot=None):
    cell_dir = os.path.join(data_directory, cell_name)
    
    # Find all .dat files in the cell directory
    data_files = [f for f in os.listdir(cell_dir) if f.endswith('.dat')]
    data_files.sort()  # Ensure files are in order
    
    all_data = []
    total_duration = 0
    config = None
    
    for data_file in data_files:
        data_path = os.path.join(cell_dir, data_file)
        xml_file = os.path.splitext(data_file)[0] + '.xml'
        xml_path = os.path.join(cell_dir, xml_file)
        
        if not os.path.exists(xml_path):
            print(f"Warning: XML file not found for {data_file}. Skipping this file.")
            continue
        
        config = parse_xml_config(xml_path)
        data = load_data(data_path, config)
        all_data.append(data)
        total_duration += data.shape[0] / config['samplingRate']
    
    if not all_data:
        print(f"No valid data files found for cell {cell_name}")
        return
    
    combined_data = np.concatenate(all_data, axis=0)
    
    if channels_to_plot is None:
        channels_to_plot = range(min(config['nChannels'], 3))  # Plot up to 3 channels by default
    
    plot_data_with_zoom(combined_data, config, channels_to_plot)
    
    # Detect spikes on the last channel (assuming it's the intracellular channel)
    intracellular_channel = config['nChannels'] - 1
    spike_times = detect_spikes(combined_data, intracellular_channel, config=config)
    
    print(f"Detected {len(spike_times)} spikes on the intracellular channel")
    print(f"Mean firing rate: {len(spike_times) / total_duration:.2f} Hz")
    
    # Compare with IntraExtra.xls data
    cell_info = dataset_info.get_cell_info(cell_name)
    if cell_info:
        print("\nComparison with IntraExtra.xls:")
        print(f"Recording time in IntraExtra.xls: {cell_info['recording time']} minutes")
        print(f"Actual recording time: {total_duration / 60:.2f} minutes")
        print(f"Number of files in IntraExtra.xls: {cell_info['# of files']}")
        print(f"Actual number of files: {len(data_files)}")
        print(f"Number of channels in IntraExtra.xls: {cell_info['nChannels']}")
        print(f"Actual number of channels: {config['nChannels']}")
    else:
        print(f"\nNo information found for cell {cell_name} in IntraExtra.xls")

# Example usage
dataset_info = DatasetInfo("IntraExtra.xls")  # or "IntraExtra.xlsx" if you have the newer format
cell_name = "d5331"
data_directory = "."  # Assuming the script is run from the parent directory of the cell folders

analyze_data(cell_name, data_directory, dataset_info, channels_to_plot=[0, -2, -1])  # Plot first channel, intracellular, and current
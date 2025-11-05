"""
WTVB01-BT50 Bluetooth Vibration Sensor - Streamlit Dashboard
This app connects to the sensor and displays real-time vibration data with time series plots,
spectrograms, and CSV export functionality.

Dependencies:
- streamlit (pip install streamlit)
- bleak (pip install bleak)
- numpy (pip install numpy)
- scipy (pip install scipy)
- pandas (pip install pandas)
- matplotlib (pip install matplotlib)

Run with:
    streamlit run vibration_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import asyncio
import threading
import struct
import time
import logging
import datetime
import os
import queue
from io import BytesIO
from bleak import BleakClient
from bleak import BleakScanner

# Configure page
st.set_page_config(
    page_title="WTVB01-BT50 Vibration Monitor",
    page_icon="ðŸ“Š",
    layout="wide"
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Buffer size for time series data
BUFFER_SIZE = 500

class WTVB01_BT50:
    """Interface for the WTVB01-BT50 vibration sensor"""
    
    def __init__(self, scale_factor=1.0):
        """Initialize the sensor interface"""
        # Vibration data (X, Y, Z)
        self.vibration_x = 0.0
        self.vibration_y = 0.0
        self.vibration_z = 0.0
        
        # Other sensor values
        self.aux_value1 = 0
        self.aux_value2 = 0
        self.aux_value3 = 0
        
        # Scaling factor
        self.scale_factor = scale_factor
        
        # Data buffers for time series
        self.timestamps = np.zeros(BUFFER_SIZE)
        self.x_buffer = np.zeros(BUFFER_SIZE)
        self.y_buffer = np.zeros(BUFFER_SIZE)
        self.z_buffer = np.zeros(BUFFER_SIZE)
        self.magnitude_buffer = np.zeros(BUFFER_SIZE)
        self.aux1_buffer = np.zeros(BUFFER_SIZE)
        self.aux2_buffer = np.zeros(BUFFER_SIZE)
        self.aux3_buffer = np.zeros(BUFFER_SIZE)
        
        # Start time
        self.start_time = time.time()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Data queue for CSV export
        self.data_queue = queue.Queue()
        
        # Event to signal when there's new data
        self.new_data_event = threading.Event()
        
        # Connected state
        self.connected = False
        
        # Sample rate - default to 5000 Hz to ensure we can display up to 2000 Hz in spectrogram
        self.sample_rate = 5000.0
    
    def process_data(self, data):
        """
        Process a data packet from the sensor
        
        Args:
            data (bytes): The raw data packet
        """
        if len(data) < 11 or data[0] != 0x55 or data[1] != 0x61:
            return False
        
        # Extract the three main data values
        data_values = []
        for i in range(3):
            val = struct.unpack('<h', data[2+i*2:4+i*2])[0]
            data_values.append(val)
        
        # Extract the extra bytes
        extra_bytes = []
        for i in range(8, min(len(data), 11)):
            extra_bytes.append(data[i])
        
        # Pad extra_bytes if needed
        while len(extra_bytes) < 3:
            extra_bytes.append(0)
        
        # Get current timestamp
        current_time = time.time() - self.start_time
        
        with self.lock:
            # Update the vibration values
            self.vibration_x = data_values[0] * self.scale_factor
            self.vibration_y = data_values[1] * self.scale_factor
            self.vibration_z = data_values[2] * self.scale_factor
            
            # Update auxiliary values
            self.aux_value1 = extra_bytes[0]
            self.aux_value2 = extra_bytes[1]
            self.aux_value3 = extra_bytes[2]
            
            # Calculate magnitude
            magnitude = np.sqrt(self.vibration_x**2 + self.vibration_y**2 + self.vibration_z**2)
            
            # Shift buffers and add new data
            self.timestamps[:-1] = self.timestamps[1:]
            self.x_buffer[:-1] = self.x_buffer[1:]
            self.y_buffer[:-1] = self.y_buffer[1:]
            self.z_buffer[:-1] = self.z_buffer[1:]
            self.magnitude_buffer[:-1] = self.magnitude_buffer[1:]
            self.aux1_buffer[:-1] = self.aux1_buffer[1:]
            self.aux2_buffer[:-1] = self.aux2_buffer[1:]
            self.aux3_buffer[:-1] = self.aux3_buffer[1:]
            
            self.timestamps[-1] = current_time
            self.x_buffer[-1] = self.vibration_x
            self.y_buffer[-1] = self.vibration_y
            self.z_buffer[-1] = self.vibration_z
            self.magnitude_buffer[-1] = magnitude
            self.aux1_buffer[-1] = self.aux_value1
            self.aux2_buffer[-1] = self.aux_value2
            self.aux3_buffer[-1] = self.aux_value3
            
            # Add data to queue for CSV export
            data_row = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                'vibration_x': self.vibration_x,
                'vibration_y': self.vibration_y,
                'vibration_z': self.vibration_z,
                'magnitude': magnitude,
                'aux_value1': self.aux_value1,
                'aux_value2': self.aux_value2,
                'aux_value3': self.aux_value3,
                'raw_data1': data_values[0],
                'raw_data2': data_values[1],
                'raw_data3': data_values[2],
                'raw_extra1': extra_bytes[0],
                'raw_extra2': extra_bytes[1],
                'raw_extra3': extra_bytes[2]
            }
            self.data_queue.put(data_row)
        
        # Signal that there's new data
        self.new_data_event.set()
        
        return True
    
    def get_data_for_plots(self):
        """Get a copy of the buffered data for plotting (thread-safe)"""
        with self.lock:
            valid_indices = np.where(self.timestamps > 0)[0]
            if len(valid_indices) == 0:
                return None
            
            return {
                'timestamps': self.timestamps[valid_indices].copy(),
                'x': self.x_buffer[valid_indices].copy(),
                'y': self.y_buffer[valid_indices].copy(),
                'z': self.z_buffer[valid_indices].copy(),
                'magnitude': self.magnitude_buffer[valid_indices].copy(),
                'aux1': self.aux1_buffer[valid_indices].copy(),
                'aux2': self.aux2_buffer[valid_indices].copy(),
                'aux3': self.aux3_buffer[valid_indices].copy()
            }
    
    def get_latest_values(self):
        """Get the latest values (thread-safe)"""
        with self.lock:
            return {
                'x': self.vibration_x,
                'y': self.vibration_y,
                'z': self.vibration_z,
                'magnitude': np.sqrt(self.vibration_x**2 + self.vibration_y**2 + self.vibration_z**2),
                'aux1': self.aux_value1,
                'aux2': self.aux_value2,
                'aux3': self.aux_value3
            }
    
    def reset_data(self):
        """Reset all data buffers"""
        with self.lock:
            self.timestamps = np.zeros(BUFFER_SIZE)
            self.x_buffer = np.zeros(BUFFER_SIZE)
            self.y_buffer = np.zeros(BUFFER_SIZE)
            self.z_buffer = np.zeros(BUFFER_SIZE)
            self.magnitude_buffer = np.zeros(BUFFER_SIZE)
            self.aux1_buffer = np.zeros(BUFFER_SIZE)
            self.aux2_buffer = np.zeros(BUFFER_SIZE)
            self.aux3_buffer = np.zeros(BUFFER_SIZE)
            self.start_time = time.time()


def notification_handler(sender, data, sensor):
    """
    Handle Bluetooth notifications
    
    Args:
        sender: The sender characteristic
        data: The data received
        sensor: The sensor object
    """
    sensor.process_data(data)


async def connect_to_sensor(sensor, device_address=None):
    """
    Connect to the sensor
    
    Args:
        sensor: The sensor object
        device_address: Optional device address or name
    
    Returns:
        client: The BleakClient object if connected, None otherwise
    """
    # Scan for devices
    logger.info("Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover()
    
    if not devices:
        logger.error("No Bluetooth devices found.")
        return None
    
    # Find the device
    device = None
    if device_address:
        # Try matching by address or name
        for d in devices:
            if (d.address and d.address.lower() == device_address.lower()) or \
               (d.name and device_address.lower() in d.name.lower()):
                device = d
                logger.info(f"Found specified device: {d.name} ({d.address})")
                break
    else:
        # Look for WitMotion devices
        for d in devices:
            if d.name and ("WTVB01-BT50" in d.name or "wit" in d.name.lower() or 
                          "jy" in d.name.lower() or "hc" in d.name.lower()):
                device = d
                logger.info(f"Found WitMotion device: {d.name} ({d.address})")
                break
    
    if not device:
        # List available devices
        logger.error("No suitable device found.")
        devices_list = []
        for i, d in enumerate(devices, 1):
            devices_list.append(f"{i}. {d.name or 'Unknown'} ({d.address})")
        return None, devices_list
    
    # Connect to the device
    logger.info(f"Connecting to {device.name or 'Unknown'} ({device.address})...")
    
    client = BleakClient(device)
    try:
        await client.connect()
        if not client.is_connected:
            logger.error("Failed to connect to the device")
            return None, []
        
        logger.info("Connected successfully!")
        
        # Create custom notification handler
        def custom_handler(sender, data):
            notification_handler(sender, data, sensor)
        
        # Find notification characteristic and subscribe
        found_characteristic = False
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties:
                    try:
                        logger.info(f"Setting up notifications on {char.uuid}")
                        await client.start_notify(char.uuid, custom_handler)
                        found_characteristic = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to set up notifications on {char.uuid}: {e}")
            
            if found_characteristic:
                break
        
        if not found_characteristic:
            # Try common UUIDs for BLE sensors
            common_uuids = [
                "0000ffe1-0000-1000-8000-00805f9b34fb",  # Common for HC-05/HC-06
                "6e400003-b5a3-f393-e0a9-e50e24dcca9e",  # Nordic UART RX
            ]
            
            for uuid in common_uuids:
                try:
                    logger.info(f"Trying common UUID: {uuid}")
                    await client.start_notify(uuid, custom_handler)
                    found_characteristic = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to set up notifications on {uuid}: {e}")
        
        if not found_characteristic:
            logger.error("Could not find a suitable characteristic for notifications")
            await client.disconnect()
            return None, []
        
        # Set connected state
        sensor.connected = True
        
        return client, []
    
    except Exception as e:
        logger.error(f"Error connecting to device: {e}")
        return None, []


async def disconnect_sensor(client):
    """
    Disconnect from the sensor
    
    Args:
        client: The BleakClient object
    """
    if client and client.is_connected:
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties:
                    try:
                        await client.stop_notify(char.uuid)
                    except:
                        pass
        
        await client.disconnect()
        logger.info("Disconnected from sensor")


def calculate_spectrogram(data, sample_rate=100.0, window_size=256, overlap=0.75):
    """
    Calculate spectrogram for the given data
    
    Args:
        data: The data array
        sample_rate: Sample rate in Hz
        window_size: Window size for the spectrogram
        overlap: Overlap factor
    
    Returns:
        Spectrogram data, frequencies, and times
    """
    if len(data) < window_size:
        return None, None, None
    
    # Make sure we have enough data for the window size
    if len(data) < window_size:
        # Pad with zeros if we don't have enough data
        data = np.pad(data, (0, window_size - len(data)), 'constant')
    
    # Use higher window size for better frequency resolution
    # Need to have this large enough to give us frequency resolution up to 2000 Hz
    nperseg = min(window_size, len(data) // 2)
    nperseg = max(nperseg, 256)  # Ensure minimum size for good resolution
    
    # Calculate spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        data,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=int(nperseg * overlap),
        scaling='density',
        mode='psd',
        detrend='constant',  # Add detrending to remove DC component
        nfft=4096  # Force larger FFT size for higher frequency resolution
    )
    
    # Convert to dB scale (log scale), add small offset to avoid log(0)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Make sure our frequency range goes up to 2000 Hz or as close as possible
    if max(frequencies) < 2000 and sample_rate > 4000:
        logger.warning(f"Maximum frequency in spectrogram is only {max(frequencies):.1f} Hz")
    
    return Sxx_db, frequencies, times


# Create a background thread for asyncio loop
def run_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# Function to run asyncio tasks
def run_async_task(task, loop):
    return asyncio.run_coroutine_threadsafe(task, loop)


# Function to create plotting figures
def create_figures(data, spectrogram_axis='x', sample_rate=100.0, y_min=None, y_max=None, 
                   freq_max=2000, window_size=256):
    """
    Create time series and spectrogram figures
    
    Args:
        data: The data dictionary containing time series data
        spectrogram_axis: Which axis to create spectrogram for ('x', 'y', 'z', or 'magnitude')
        sample_rate: Sample rate in Hz
        y_min: Minimum y-axis value for time series plot
        y_max: Maximum y-axis value for time series plot
        freq_max: Maximum frequency to display in spectrogram
        window_size: Window size for spectrogram calculation
    
    Returns:
        Time series figure and spectrogram figure
    """
    if data is None:
        return None, None
    
    # Create time series figure
    fig_timeseries = plt.figure(figsize=(10, 4))
    ax = fig_timeseries.add_subplot(111)
    ax.plot(data['timestamps'], data['x'], 'r-', label='X')
    ax.plot(data['timestamps'], data['y'], 'g-', label='Y')
    ax.plot(data['timestamps'], data['z'], 'b-', label='Z')
    ax.plot(data['timestamps'], data['magnitude'], 'k-', label='Magnitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vibration Amplitude')
    ax.set_title('Vibration Time Series')
    ax.grid(True)
    ax.legend()
    
    # Apply manual y-axis scaling if provided
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    
    # Create spectrogram figure for the selected axis
    axis_data = data[spectrogram_axis.lower()]
    
    # Modify window size to ensure better frequency resolution
    # Make window size a power of 2 for efficient FFT
    window_size = max(256, window_size)
    
    Sxx, freq, times = calculate_spectrogram(axis_data, sample_rate, window_size=window_size)
    
    if Sxx is not None:
        fig_spec = plt.figure(figsize=(10, 4))
        ax = fig_spec.add_subplot(111)
        im = ax.pcolormesh(times, freq, Sxx, shading='gouraud', cmap=cm.viridis)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'{spectrogram_axis.upper()}-axis Vibration Spectrogram')
        
        # Explicitly force y-axis to be 0-2000 Hz regardless of data
        ax.set_ylim(0, 2000)
        
        # Also set ticks to emphasize the range
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        
        # Add text annotation to confirm range
        ax.text(0.02, 0.98, f'Range: 0-2000 Hz', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
        fig_spec.colorbar(im, ax=ax, label='Power Spectral Density (dB)')
    else:
        fig_spec = None
    
    return fig_timeseries, fig_spec


# Function to export data to CSV
def export_data_to_csv(data_queue, stop_event, filename):
    """
    Export data to CSV file
    
    Args:
        data_queue: Queue containing data rows
        stop_event: Event to signal when to stop
        filename: Output filename
    """
    if not filename:
        return
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create CSV file and write header
    with open(filename, 'w', newline='') as f:
        f.write("Timestamp,Vibration_X,Vibration_Y,Vibration_Z,Magnitude,Aux1,Aux2,Aux3,Raw_Data1,Raw_Data2,Raw_Data3,Raw_Extra1,Raw_Extra2,Raw_Extra3\n")
    
    # Write data rows as they come in
    while not stop_event.is_set():
        try:
            data_row = data_queue.get(timeout=0.1)
            with open(filename, 'a', newline='') as f:
                f.write(f"{data_row['timestamp']},{data_row['vibration_x']},{data_row['vibration_y']},{data_row['vibration_z']},{data_row['magnitude']},{data_row['aux_value1']},{data_row['aux_value2']},{data_row['aux_value3']},{data_row['raw_data1']},{data_row['raw_data2']},{data_row['raw_data3']},{data_row['raw_extra1']},{data_row['raw_extra2']},{data_row['raw_extra3']}\n")
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")


# Main app
def main():
    # Set up event loop in a background thread
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    thread.start()
    
    # Set title
    st.title("WTVB01-BT50 Vibration Sensor Dashboard")
    
    # Create sensor object
    if 'sensor' not in st.session_state:
        st.session_state.sensor = WTVB01_BT50()
        st.session_state.client = None
        st.session_state.logging = False
        st.session_state.log_thread = None
        st.session_state.stop_logging = threading.Event()
        st.session_state.data_export_filename = "vibration_data.csv"
        st.session_state.spectrogram_axis = "x"  # Default to X-axis spectrogram
        # Default scaling values - set to None for auto-scaling by default
        st.session_state.y_min = None 
        st.session_state.y_max = None
        st.session_state.freq_max = 2000  # Default to 2000 Hz
        st.session_state.window_size = 256
    
    # Sidebar for connection controls
    with st.sidebar:
        st.header("Sensor Connection")
        
        device_address = st.text_input("Device Address or Name (optional)", 
                                       placeholder="e.g., 00:11:22:33:44:55 or WTVB01")
        
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.sensor.connected:
                if st.button("Connect", use_container_width=True):
                    with st.spinner("Connecting to sensor..."):
                        client, devices_list = run_async_task(
                            connect_to_sensor(st.session_state.sensor, device_address),
                            loop
                        ).result()
                        
                        if client:
                            st.session_state.client = client
                            st.success("Connected!")
                        else:
                            st.error("Failed to connect")
                            if devices_list:
                                st.write("Available devices:")
                                for device in devices_list:
                                    st.write(device)
            else:
                if st.button("Disconnect", use_container_width=True):
                    with st.spinner("Disconnecting..."):
                        if st.session_state.client:
                            run_async_task(
                                disconnect_sensor(st.session_state.client),
                                loop
                            ).result()
                            st.session_state.client = None
                            st.session_state.sensor.connected = False
                            st.session_state.sensor.reset_data()
                    
                    st.experimental_rerun()
        
        with col2:
            if st.button("Reset Data", use_container_width=True):
                st.session_state.sensor.reset_data()
        
        st.header("Data Export")
        
        # Download current data
        col1, col2 = st.columns(2)
        with col1:
            # Create a download button for the current data
            if st.button("Download Current Data", use_container_width=True, key="download_current_data_button"):
                data = st.session_state.sensor.get_data_for_plots()
                if data is not None:
                    # Convert to DataFrame
                    df = pd.DataFrame({
                        'Timestamp': data['timestamps'],
                        'Vibration_X': data['x'],
                        'Vibration_Y': data['y'],
                        'Vibration_Z': data['z'],
                        'Magnitude': data['magnitude'],
                        'Aux1': data['aux1'],
                        'Aux2': data['aux2'],
                        'Aux3': data['aux3']
                    })
                    
                    # Convert to CSV
                    csv = df.to_csv(index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="vibration_data_export.csv",
                        mime="text/csv",
                        key="download_current_csv_button"
                    )
                else:
                    st.warning("No data to download")
        
        with col2:
            # Copy to clipboard button
            if st.button("Copy Settings to Clipboard", use_container_width=True, key="copy_settings_button"):
                # Create a string with current settings
                settings_str = f"""
                Sample Rate: {sample_rate} Hz
                Y-axis Range: {st.session_state.y_min} to {st.session_state.y_max}
                Selected Axis: {st.session_state.spectrogram_axis}
                Max Frequency: {st.session_state.freq_max} Hz
                Window Size: {st.session_state.window_size}
                """
                # Use JavaScript to copy to clipboard
                st.markdown(
                    f"""
                    <script>
                    const copyToClipboard = str => {{
                        const el = document.createElement('textarea');
                        el.value = str;
                        document.body.appendChild(el);
                        el.select();
                        document.execCommand('copy');
                        document.body.removeChild(el);
                    }};
                    copyToClipboard(`{settings_str}`);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.success("Settings copied to clipboard!")
        
        # CSV export options
        st.header("Data Logging")
        st.session_state.data_export_filename = st.text_input(
            "CSV Export Filename", 
            value=st.session_state.data_export_filename,
            placeholder="vibration_data.csv"
        )
        
        csv_col1, csv_col2 = st.columns(2)
        with csv_col1:
            if not st.session_state.logging:
                if st.button("Start Logging", use_container_width=True, key="start_logging_button"):
                    if st.session_state.data_export_filename:
                        st.session_state.stop_logging.clear()
                        st.session_state.log_thread = threading.Thread(
                            target=export_data_to_csv,
                            args=(
                                st.session_state.sensor.data_queue,
                                st.session_state.stop_logging,
                                st.session_state.data_export_filename
                            ),
                            daemon=True
                        )
                        st.session_state.log_thread.start()
                        st.session_state.logging = True
                        st.success(f"Logging to {st.session_state.data_export_filename}")
                    else:
                        st.error("Please enter a filename")
            else:
                if st.button("Stop Logging", use_container_width=True, key="stop_logging_button"):
                    st.session_state.stop_logging.set()
                    if st.session_state.log_thread:
                        st.session_state.log_thread.join(timeout=1.0)
                    st.session_state.logging = False
                    st.info("Logging stopped")
        
        with csv_col2:
            # Create a download button for the current data
            if st.button("Download Current Data", use_container_width=True, key="download_data_button"):
                data = st.session_state.sensor.get_data_for_plots()
                if data is not None:
                    # Convert to DataFrame
                    df = pd.DataFrame({
                        'Timestamp': data['timestamps'],
                        'Vibration_X': data['x'],
                        'Vibration_Y': data['y'],
                        'Vibration_Z': data['z'],
                        'Magnitude': data['magnitude'],
                        'Aux1': data['aux1'],
                        'Aux2': data['aux2'],
                        'Aux3': data['aux3']
                    })
                    
                    # Convert to CSV
                    csv = df.to_csv(index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="vibration_data_export.csv",
                        mime="text/csv",
                        key="download_csv_button"
                    )
                else:
                    st.warning("No data to download")
        
        # Display logging status
        if st.session_state.logging:
            st.info(f"Logging to: {st.session_state.data_export_filename}")
        
        # Sample rate setting
        st.header("Settings")
        
        # Main display settings
        sample_rate = st.slider("Sample Rate (Hz)", 1000, 10000, 5000, 
                               help="Higher sample rate allows detecting higher frequencies")
        st.session_state.sensor.sample_rate = sample_rate
        
        # Time series plot scaling
        st.subheader("Time Series Scaling")
        col1, col2 = st.columns(2)
        
        # Auto-scale option - default to checked
        auto_scale = st.checkbox("Auto-scale Y-axis", value=True)
        
        if not auto_scale:
            with col1:
                st.session_state.y_min = st.number_input("Y-axis Min", value=-1500 if st.session_state.y_min is None else st.session_state.y_min)
            with col2:
                st.session_state.y_max = st.number_input("Y-axis Max", value=1500 if st.session_state.y_max is None else st.session_state.y_max)
        else:
            st.session_state.y_min = None
            st.session_state.y_max = None
        
        # Spectrogram settings
        st.subheader("Spectrogram Settings")
        
        # Spectrogram axis selection
        st.session_state.spectrogram_axis = st.selectbox(
            "Spectrogram Axis",
            options=["X", "Y", "Z", "Magnitude"],
            index=0  # Default to X
        )
        
        # Frequency range for spectrogram - fixed at 2000 Hz
        st.session_state.freq_max = 2000  # Fixed to 2000 Hz
        st.write("Frequency Range: 0-2000 Hz")
        
        # Add note about frequency range
        st.info("The spectrogram displays frequencies from 0 to 2000 Hz, which is ideal for motor/fan vibration analysis.")
        
        # Window size for spectrogram (controls frequency resolution)
        window_sizes = [64, 128, 256, 512, 1024]
        resolution_options = ["Very Low", "Low", "Medium", "High", "Very High"]
        
        # Find current window size index
        if st.session_state.window_size in window_sizes:
            current_index = window_sizes.index(st.session_state.window_size)
        else:
            current_index = 2  # Default to medium
        
        selected_resolution = st.selectbox(
            "Spectrogram Resolution",
            options=resolution_options,
            index=current_index,
            help="Higher resolution gives better frequency detail but requires more data"
        )
        # Convert the text selection back to window size
        st.session_state.window_size = window_sizes[resolution_options.index(selected_resolution)]
        
        # Display connection status
        st.header("Status")
        if st.session_state.sensor.connected:
            st.success("Connected")
        else:
            st.warning("Disconnected")
    
    # Main content
    # Current values
    current_values = st.session_state.sensor.get_latest_values()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("X-axis", f"{current_values['x']:.1f}")
    with col2:
        st.metric("Y-axis", f"{current_values['y']:.1f}")
    with col3:
        st.metric("Z-axis", f"{current_values['z']:.1f}")
    with col4:
        st.metric("Magnitude", f"{current_values['magnitude']:.1f}")
    
    # Copy values button
    if st.button("Copy Current Values", key="copy_values_button"):
        values_text = f"X: {current_values['x']:.1f}, Y: {current_values['y']:.1f}, Z: {current_values['z']:.1f}, Magnitude: {current_values['magnitude']:.1f}"
        st.markdown(
            f"""
            <script>
            navigator.clipboard.writeText(`{values_text}`);
            </script>
            """,
            unsafe_allow_html=True
        )
        st.success(f"Values copied: {values_text}")
    
    # Time series plot
    st.subheader("Vibration Time Series")
    timeseries_placeholder = st.empty()
    
    # Full-width spectrogram
    st.subheader(f"{st.session_state.spectrogram_axis}-axis Spectrogram")
    spectrogram_placeholder = st.empty()
    
    # Placeholder for refresh status
    refresh_placeholder = st.empty()
    
    # Update loop for the plots
    while True:
        if st.session_state.sensor.new_data_event.is_set():
            st.session_state.sensor.new_data_event.clear()
            
            # Get the latest data
            data = st.session_state.sensor.get_data_for_plots()
            
            # Create plots
            fig_timeseries, fig_spec = create_figures(
                data, 
                spectrogram_axis=st.session_state.spectrogram_axis.lower(),
                sample_rate=st.session_state.sensor.sample_rate,  # Use the sensor's sample rate
                y_min=st.session_state.y_min,
                y_max=st.session_state.y_max,
                freq_max=st.session_state.freq_max,
                window_size=st.session_state.window_size
            )
            
            # Update plots
            if fig_timeseries:
                timeseries_placeholder.pyplot(fig_timeseries)
                plt.close(fig_timeseries)
            
            if fig_spec:
                spectrogram_placeholder.pyplot(fig_spec)
                plt.close(fig_spec)
            
            # Update refresh status
            refresh_time = datetime.datetime.now().strftime("%H:%M:%S")
            refresh_placeholder.info(f"Last updated: {refresh_time}")
        
        # Sleep briefly to avoid high CPU usage
        time.sleep(0.1)


if __name__ == "__main__":
    main()

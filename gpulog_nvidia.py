import subprocess
import time

# Set the log file name
log_file = 'gpu_usage.log'

# Run resnet.py
resnet_process = subprocess.Popen(['python3', 'resnet.py'])

try:
    # Open the log file in append mode
    with open(log_file, 'a') as log:
        while resnet_process.poll() is None:  # Check if the resnet.py process is still running
            # Get the current time
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Use the nvidia-smi command to get information about all GPUs
            nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
            gpu_info_lines = nvidia_smi_output.decode('utf-8').strip().split('\n')
            
            for line in gpu_info_lines:
                # Parse information for each GPU
                gpu_info = line.split(',')
                gpu_index = gpu_info[0]
                gpu_memory_used = gpu_info[1]
                gpu_memory_total = gpu_info[2]
                gpu_utilization = gpu_info[3]
                
                # Record time, GPU index, GPU memory usage, and utilization to the log file
                log_entry = f'{current_time}, GPU{gpu_index} Memory Used: {gpu_memory_used} MB, GPU{gpu_index} Memory Total: {gpu_memory_total} MB, GPU{gpu_index} Utilization: {gpu_utilization}%\n'
                log.write(log_entry)
            
            # Record every 1 seconds
            time.sleep(1)

except KeyboardInterrupt:
    # Stop monitoring and close the log file if the user presses Ctrl+C
    print("Monitoring stopped.")
except Exception as e:
    # Catch other exceptions and print the error message
    print(f"An error occurred: {str(e)}")

# Wait for the resnet.py process to complete
resnet_process.wait()

# Close the log file
log.close()

import subprocess
import time
import json

# Set the log file name
log_file = 'gpu_usage.log'

# Run matrix_mul.py
resnet_process = subprocess.Popen(['python3', 'matrix_mul.py'])

try:
    # Open the log file in append mode
    with open(log_file, 'a') as log:
        while resnet_process.poll() is None:  # Check if the matrix_mul.py process is still running
            # Get the current time
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Use the rocm-smi command to get GPU information (Make sure ROCm is installed and environment variables are set)
            rocm_smi_output = subprocess.check_output(['rocm-smi', '--showmemuse', '--showuse', '--json']).decode('utf-8')
            
            # Parse JSON-formatted GPU information
            gpu_info = json.loads(rocm_smi_output)
            
            for gpu_index, gpu_data in gpu_info.items():
                gpu_memory_used = gpu_data["GPU memory use (%)"]
                gpu_memory_activity = gpu_data["Memory Activity"]
                gpu_utilization = gpu_data["GPU use (%)"]
                
                # Record time, GPU index, GPU memory usage, memory activity, and utilization to the log file
                log_entry = f'{current_time}, GPU{gpu_index} Memory Used: {gpu_memory_used}%, Memory Activity: {gpu_memory_activity} KB, GPU Utilization: {gpu_utilization}%\n'
                log.write(log_entry)
            
            # Record every 1 seconds
            time.sleep(1)

except KeyboardInterrupt:
    # Stop monitoring and close the log file if the user presses Ctrl+C
    print("Monitoring stopped.")
except Exception as e:
    # Catch other exceptions and print the error message
    print(f"An error occurred: {str(e)}")

# Wait for the matrix_mul.py process to complete
resnet_process.wait()

# Close the log file
log.close()

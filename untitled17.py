# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:24:28 2024

@author: starv
"""

import os
import re
from datetime import datetime, timedelta

def get_file_name(file_path):
    return os.path.basename(file_path)

def extract_start_time(file_name, time_format='%Y%m%d_%H%M%S'):
    # Adjust the regular expression to match the new time format 'YYYYMMDD_HHMMSS'
    match = re.search(r'\d{8}_\d{6}', file_name)
    if match:
        start_time_str = match.group()
        start_time = datetime.strptime(start_time_str, time_format)
        return start_time
    else:
        raise ValueError("No valid start time found in file name")

def generate_hourly_ticks(start_time, duration_hours=6):
    ticks = [start_time + timedelta(hours=i) for i in range(duration_hours + 1)]
    return ticks

def main():
    # Example file path
    file_path = r'C:\path\to\your\directory\file_20240521_120000.txt'
    
    try:
        file_name = get_file_name(file_path)
        start_time = extract_start_time(file_name)
        print(f"Start time extracted from file name: {start_time}")
        
        hourly_ticks = generate_hourly_ticks(start_time)
        print("Hourly ticks:")
        for tick in hourly_ticks:
            print(tick)
    
    except (FileNotFoundError, ValueError) as e:
        print(e)

if __name__ == "__main__":
    main()
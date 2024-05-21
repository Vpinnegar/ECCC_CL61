# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:13:29 2024

@author: starv
"""

import os
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt # import libraries
import pandas as pd # import libraries
import netCDF4 as nc # import libraries
import numpy as np
from matplotlib import cm, ticker
from numpy import ma
import matplotlib.style as mplstyle
import matplotlib
from matplotlib import colors
import scipy


directory = "C:/Users/starv/OneDrive/Desktop/Code/Pasto/CL61/"
# print(os.listdir(directory))
# Get a list of files in the directory
First = []
second = []
third = []
fourth = []
files = [f for f in os.listdir(directory) if f.endswith('.nc')] 
dis = np.linspace(0,120,120)
for im in range(len(dis)):
    current_figure_index = im
    
    ## 32,33,34 is not there
    current_file = files[current_figure_index]
    # plt.title(current_file)
    nfile = nc.Dataset(directory + current_file)
    
    # nfile = nc.Dataset("C:\Users\starv\Downloads\EastgateCL51_20230821_222043.nc")
    # current_file = "C:/Users/starv/OneDrive/Desktop/Code/Pasto/EastgateCL61_20230820_222001.nc"
    # plt.title(current_file)
    # nfile = nc.Dataset(current_file)
    # # plt.subplot(2,1,1)
    # print(nfile)
    # levels = np.linspace(start = 0, stop = 0.00007, num = 100)
    backspol = nfile['p_pol'][:]
    backsxpol = nfile['x_pol'][:]
    bScatatt= nfile['beta_att'][:]
    clouds = nfile['sky_condition_cloud_layer_heights'][:]
    vd = backsxpol/(backspol + backsxpol)
    print(nfile)
    
    noise1 = nfile['beta_att_noise_level'][:]
    
    
    bave = bScatatt[0:20,:]
    bavve = np.mean(bave, axis=0)
    ten_profiles = (bScatatt[0:20,:] - bave)**2
    m = 5
    
    
    profiles = bave / np.sqrt((1/(2*m))*(np.sum(ten_profiles, axis=0)))
    plt.figure()
    plt.plot(profiles)
    plt.show()
    
    # mask1 = profiles < 1
    # bave[mask1] = 0
    plt.figure()
    plt.plot(bave)
    plt.show()
    
    
    def calculate_snr_profiles(data):
        num_profiles = data.shape[0]
        num_samples = data.shape[1]
        
        snr_profiles = np.zeros((num_profiles, num_samples))
        
        for i in range(num_profiles):
            # Compute indices for the ten profiles
            start_index = max(0, i - 5)
            end_index = min(num_profiles, i + 5)
            
            top = np.mean(bScatatt[start_index:end_index,:], axis=0)
            ten_profiles = (bScatatt[start_index:end_index,:] - top)**2
            # Average the ten profiles
            avg_ten_profiles = np.sum(ten_profiles, axis=0)
            
            # Compute the profile divided by the square root of the averaged values of ten profiles
            profile_divided_by_sqrt_avg_ten_profiles = top / np.sqrt((1/(2*m))*(avg_ten_profiles))
            
            # Compute the square of the original profile
            # profile_squared = data[i] ** 2
            
            # Subtract the squared original profile from the result obtained above
            # result = profile_divided_by_sqrt_avg_ten_profiles - profile_squared
            
            # Take the mean of the result
            # mean_result = np.mean(result)
            
            # Calculate SNR
            snr_profiles[i] = profile_divided_by_sqrt_avg_ten_profiles
            
        return snr_profiles
        
    tal = calculate_snr_profiles(bScatatt)
    levels = np.linspace(start = 0, stop = 5, num = 300)
    # plt.figure(figsize=(10,4))    
    # plt.yticks([0,1000,2000,3000],['0','5','10','15'])
    # # cb.set_yticks([0,100,200,300,399])
    # # plt.yticklabels()
    # plt.ylabel("Altitude (Km)")
    
    
    plt.figure(figsize=(10,4))    
    plt.yticks([0,1000,2000,3000],['0','5','10','15'])
    plt.ylabel("Altitude (Km)")
    plt.contourf(np.ndarray.transpose(tal),levels , cmap= 'turbo')
    plt.xticks([0,60,120,180,240,300,359],['10','11','12','13','14','15','16'])
    plt.xlabel('Time (UTC)')
    plt.colorbar()
    plt.savefig("C:/Users/starv/OneDrive/Desktop/VictoriaPhD/" + current_file + "SNR.png", dpi=800)
    plt.show()
    
    
    plt.figure()
    levels = np.linspace(start = 0, stop = 0.00008, num = 500)
    plt.figure(figsize=(10,4))    
    plt.yticks([0,1000,2000,3000],['0','5','10','15'])
    # cb.set_yticks([0,100,200,300,399])
    # plt.yticklabels()
    plt.ylabel("Altitude (Km)")
    plt.contourf(np.ndarray.transpose(bScatatt),levels, locator=ticker.LogLocator(), cmap= 'turbo')
    plt.title(current_file)
    # plt.scatter(xr,fin5, color = 'black')
    plt.xticks([0,60,120,180,240,300,360],['10','11','12','13','14','15','16'])
    plt.xlabel('Time (UTC)')
    plt.savefig("C:/Users/starv/OneDrive/Desktop/VictoriaPhD/" + current_file + ".png", dpi=800)
    plt.show()
    
    
    
    
    
    
    threshold = 0.4925
    threshold2 = 0.20
    threshold3 = 0.10
    threshold4 = 0.023
    mask2 = vd > threshold2
    
    # bScatatt = np.nan_to_num(bScatatt, nan=np.nanmean(bScatatt))
    fin = []
    fin0 = []
    fin1 = []
    fin2 = []
    fin3 = []
    fin5 = []
    window = 20
    
        
        
        
    bavs = np.convolve(bScatatt[0,:], np.ones(window)/window, mode='valid')
    
    he = np.linspace(0,3000,len(bavs))
    plt.figure()
    plt.plot(bavs,he)
    plt.yticks([0,1000,2000,3000],['0','5','10','15'])
    # cb.set_yticks([0,100,200,300,399])
    # plt.yticklabels()
    plt.ylabel("Altitude (Km)")
    plt.xlabel("Backscatter (1/m*sr)")
    plt.show()
    fl = np.linspace(0,3000,15000)
    hieght = np.linspace(0,15000,3000)
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list
    maskk = []
    for i in range(len(bScatatt)):
        # resha = backspol[i,:].reshape(-1,6)
        # aves = np.mean(resha, axis=1)
        # backspol1 = aves
        Ar = []
        # resha = backsxpol[i,:].reshape(-1,6)
        # aves = np.mean(resha, axis=1)
        # backsxpol1 = aves
        
        # sos = scipy.signal.iirfilter(4, Wn=[0.1, 2.5], fs=fs, btype="bandpass",
                                     # ftype="butter", output="sos")
        # yfilt = scipy.signal.sosfilt(sos, backspol)
        # start = 0
        # end = 300
        # aver = np.mean(bScatatt[i,start:end])
        # thres = 6e-6
        # if aver > thres:
        #     maskk.append(1.1)
        # else:
        #     maskk.append(0)
            
            
            
            
            
        # for iic in range(len(clouds[:,0])): 
            # if int(fl[clouds[i-1,0]]) > 1000:
            #     start = 0
            #     end = 1000
            #     aver = np.mean(backs1[i,start:end])
            #     precip.append(aver)
            #     pass
        if 0 < int(clouds.data[i,0]) < 3500:
            print('CORRECT')
            start = 0
            end = int(clouds[i,0])
            aver = np.mean(bScatatt[i,start:end])
            thres = 0.9e-6
            if np.any(bScatatt[i,start:end] > 0.6e-6):
                if aver > thres:
                    maskk.append(1)
                else:
                    maskk.append(0)
            else:
                maskk.append(0)
            # precip.append(aver)
            
        else: 
            start = 0
            end = 500
            thres = 0.95e-6
            aver = np.mean(bScatatt[i,start:end])
            if np.any(bScatatt[i,start:end] > 0.6e-6):
                if aver > thres:
                    maskk.append(1)
                else:
                    maskk.append(0)
            else:
                maskk.append(0)
                # precip.append(aver)
        mask1 = tal < 1
        bScatatt[mask2] = 0
        bScatatt[mask1] = 0
        backspol1 = np.convolve(bScatatt[i,:], np.ones(window)/window, mode='valid')
        backsxpol1 = np.convolve(bScatatt[i,:], np.ones(window)/window, mode='valid')
        
        # peaks, props = scipy.signal.find_peaks(yfilt, distance=0.35*fs, height=0.0)
        cwt_peaks = scipy.signal.find_peaks_cwt(backspol1, widths=np.arange(20, 400))
        # plt.figure()
        # plt.plot(backspol1[:], label='Co-pol')
        # plt.plot(backsxpol1[:], label='Cross-pol')
        
        # plt.axvline(cwt_peaks[0], color = 'b', label = 'axvline - full height')
        # plt.axvline(cwt_peaks[1], color = 'b', label = 'axvline - full height')
        # plt.axvline(cwt_peaks[2], color = 'b', label = '')
        # plt.axvline(cwt_peaks[3], color = 'b', label = 'peak')
        Ar.append(cwt_peaks[0:3])
        if len(cwt_peaks) >= 4:
            if 9e-6 > bScatatt[i, cwt_peaks[0]] > 1e-6:
                fin0.append(cwt_peaks[0])
            else:
                fin0.append(np.nan)
            if 9e-6 > bScatatt[i, cwt_peaks[1]] > 1e-6:
                fin1.append(cwt_peaks[1])
            else:
                fin1.append(np.nan)
            if 9e-6 > bScatatt[i, cwt_peaks[2]] >1e-6:
                fin2.append(cwt_peaks[2])
            else:
                fin2.append(np.nan)
            # print('3')
            if 9e-6 > bScatatt[i, cwt_peaks[3]] > 1e-6:
                fin3.append(cwt_peaks[3])
            else:
                fin3.append(np.nan)
            # print('4',end="\r")
        elif len(cwt_peaks) == 3:
            if 9e-6 > bScatatt[i, cwt_peaks[0]] > 1e-6:
                fin0.append(cwt_peaks[0])
            else:
                fin0.append(np.nan)
            if 9e-6 > bScatatt[i, cwt_peaks[1]] > 1e-6:
                fin1.append(cwt_peaks[1])
            else:
                fin1.append(np.nan)
            if 9e-6 > bScatatt[i, cwt_peaks[2]] > 1e-6:
                fin2.append(cwt_peaks[2])
            else:
                fin2.append(np.nan)
            fin3.append(np.nan)
            # print('3', end="\r")
        elif len(cwt_peaks) == 2:
            
            if 9e-6 > bScatatt[i, cwt_peaks[0]] > 1e-6:
                fin0.append(cwt_peaks[0])
            else:
                fin0.append(np.nan)
            if 9e-6 > bScatatt[i, cwt_peaks[1]] > 1e-6:
                fin1.append(cwt_peaks[1])
            else:
                fin1.append(np.nan)
            fin2.append(np.nan)
            fin3.append(np.nan)
            # print('2',end="\r")
        elif len(cwt_peaks) == 1:
            
            if 9e-6 > bScatatt[i, cwt_peaks[0]] > 1e-6:
                fin0.append(cwt_peaks[0])
            else:
                fin0.append(np.nan)
            fin1.append(np.nan)
            fin2.append(np.nan)
            fin3.append(np.nan)
            # print('1',end="\r")
        else:
            
            fin0.append(np.nan)
            fin1.append(np.nan)
            fin2.append(np.nan)
            fin3.append(np.nan)
            print('0',end="\r")
        print(i, end="\r")
        # Initialize a flag variable
        # second_if_executed = False
        # # if len(cwt_peaks) >= 2:
        # if 9e-6 > bScatatt[i, cwt_peaks[1]] > 1e-6:
        #     fin3.append(cwt_peaks[1])
        # else:
        #     fin3.append(0)
                
        # else:
        #     fin3.append(0)
        #     fin4.append(0)
        #     second_if_executed = True
        
        # if len(cwt_peaks) >= 3:
        # if 9e-6 > bScatatt[i, cwt_peaks[2]] > 1e-6:
        #     fin4.append(cwt_peaks[2])
        # else:
        #     fin4.append(0)
            
        
        # if backspol1[cwt_peaks[3]] > 1.6e-6:
        #     fin5.append(cwt_peaks[3])
        # else:
        #     fin5.append(0)  
        # fin3.append(cwt_peaks[1])
        # fin4.append(cwt_peaks[2])
        # fin5.append(cwt_peaks[3])
        # plt.plot(bScatatt[i,:], label = 'Backscatter')
        # plt.xlim(400,2500)
        # plt.legend()
        # plt.show()
        fin.append(Ar)
        # fin1 = flatten_extend(fin)
    
    
    he = np.arange(0,15000,5)
    xr = np.linspace(0,360,360)
    # fin1 = flatten_extend(fin)
    
    # for i in range(len(bScatatt)):
    # fin.append(Ar)
    # fin1 = flatten_extend(fin)
    # fin2 = flatten_extend(fin1)
    
    # max_length = max(len(arr) for arr in fin1)
    
    # Initialize a 2D NumPy array with the appropriate shape
    # result = np.zeros((360, max_length))
    
    # Fill the 2D array with values from the original list
    # for i, arr in enumerate(fin1):
        # result[i, :len(arr)] = arr
    # if len(cwt_peaks) >= 1:
    # for ip in range(len(cwt_peaks)): 
    # if ip == 0:
    #     if  9e-6 > bScatatt[i,fin1[:,0]] > 1e-6:
    #         fin0.append(cwt_peaks[ip])
    #         if ip == 1:
    #             fin1.append(cwt_peaks[ip])
    #         if ip == 2:
    #             fin2.append(cwt_peaks[ip])
    #     else:
    #         if ip == 0:
    #             fin0.append(0)
    #         if ip == 1:
    #             fin1.append(0)
    #         if ip == 2:
    #             fin2.append(0)
    
    masss = np.array(maskk)
    fin00 = np.array(fin0)
    fin10 = np.array(fin1)
    fin20 = np.array(fin2)
    fin30 = np.array(fin3)
    jj = masss >= 1
    fin00[jj] = np.nan 
    fin10[jj] = np.nan 
    fin20[jj] = np.nan 
    fin30[jj] = np.nan
    
    mil0 = fin00 > 1250
    mil1 = fin10 > 1250
    mil2 = fin20 > 1250
    mil3 = fin30 > 1250
    
    fin000 = []
    fin000.append(fin00)
    fin010 = []
    fin010.append(fin10)
    fin020 = []
    fin020.append(fin20)
    fin030 = []
    fin030.append(fin30)
    
    fin00[mil0] = np.nan
    fin10[mil1] = np.nan
    fin20[mil2] = np.nan
    fin30[mil3] = np.nan
    mil0 = fin00 < 1250
    mil1 = fin10 < 1250
    mil2 = fin20 < 1250
    mil3 = fin30 < 1250
    fin000 = np.array(fin000)
    fin010 = np.array(fin010)
    fin020 = np.array(fin020)
    fin030 = np.array(fin030)
    fin000[0,mil0] = np.nan
    fin010[0,mil1] = np.nan
    fin020[0,mil2] = np.nan
    fin030[0,mil3] = np.nan
    
    xr = np.linspace(0,len(fin0),len(fin0))
    plt.figure()
    levels = np.linspace(start = 0, stop = 0.00008, num = 500)
    plt.figure(figsize=(10,4))    
    plt.yticks([0,1000,2000,3000],['0','5','10','15'])
    # cb.set_yticks([0,100,200,300,399])
    # plt.yticklabels()
    plt.ylabel("Altitude (Km)")
    plt.contourf(np.ndarray.transpose(bScatatt),levels, norm=colors.LogNorm(), cmap= 'turbo')
    # plt.colorbar()
    plt.scatter(xr,fin00, color = 'black')
    plt.scatter(xr,fin10, color = 'blue')
    plt.scatter(xr,fin20, color = 'red')
    plt.scatter(xr,fin30, color = 'black')
    
    plt.scatter(xr,fin000[0,:], color = 'black', marker='x')
    plt.scatter(xr,fin010[0,:], color = 'blue', marker='x')
    plt.scatter(xr,fin020[0,:], color = 'red', marker='x')
    plt.scatter(xr,fin030[0,:], color = 'black', marker='x')
    plt.title(current_file)
    
    # plt.scatter(xr,fin5, color = 'black')
    plt.xticks([0,60,120,180,240,300,360],['10','11','12','13','14','15','16'])
    plt.xlabel('Time (UTC)')
    plt.savefig("C:/Users/starv/OneDrive/Desktop/VictoriaPhD/" + current_file + "LAYER.png", dpi=800)
    plt.show()



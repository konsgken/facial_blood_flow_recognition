clear all
close all
clc

% This code creates the avg image (1 image/30 frames), so we have 1 avg
% image/ sec

% Define video paths
%video_path = 'Datasets\June2019\Cropped_Videos\';
%video_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead\';
video_path = 'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead\';
%subfolders = dir(video_path);
day_subfolders = ['Day1\'; 'Day2\'; 'Day3\'];
day_len = size(day_subfolders, 1);
person_subfolders = ['01\';'02\';'03\';'04\';'05\'];
person_len = size(person_subfolders, 1);

% Define out video path
%out_video_path = 'Datasets\June2019\Amplified_Blood_Flow_Without_Attenuation\';
% out_video_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Blood_Flow_After_Attenuation\Forehead\';
%out_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead_Avg_Images_RGB\';
out_path = 'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead_Avg_Images_Gray\';

alpha = 120;
level = 4;
fl = 0.83;
fh = 1;
samplingRate = 30;
chromAttenuation = 1;

% For every day 
for i=1:day_len
    
    % For every person
    for j=1:person_len
        
        % Define video path
        vpath = [video_path day_subfolders(i, :) person_subfolders(j, :)];
        files = dir(vpath);
       
        % For every video in vpath
        for k=3:length(files)
            
            % Load video
            vfilename = [vpath files(k).name];
            v = VideoReader(vfilename);
            nframes = v.duration * v.framerate
            
            n = 1; % how many avg frames for this video
            for s=1:30:nframes                
                
                % Extract frame
                sum_frame = readFrame(v);
                
                for f=1:29
                    f
                    if hasFrame(v)
                        frame = readFrame(v);                    
                        sum_frame = sum_frame + frame;  % add new frame to sum
                    else
                        break;
                    end
                end
                
                avg_frame = sum_frame./30;
                norm_avg_frame = rgb2gray(uint8(255*mat2gray(avg_frame)));
                %norm_avg_frame = uint8(255*mat2gray(avg_frame));
                
                image_filename = [out_path day_subfolders(i, :) person_subfolders(j, :) 'day' num2str(i) '_video' num2str(k-2) '_avg_image_' num2str(n) '.jpg']
                 imwrite(norm_avg_frame, image_filename);
                
                n = n + 1;
                                
            end            
        end
    end
end
            

clear all
close all
clc

% This code creates the avg image (1 image/30 frames), so we have 1 avg
% image/ sec
counter_n=0;
% Define video paths
%video_path = 'Datasets\June2019\Cropped_Videos\';
%video_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead\';
video_path = 'C:\Users\konsg\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead\';
%subfolders = dir(video_path);
day_subfolders = ['Day1\'; 'Day2\'; 'Day3\'];
day_len = size(day_subfolders, 1);
person_subfolders = ['01\';'02\';'03\';'04\';'05\';'06\';'07\';'08\';'09\';'10\';'11\';'12\'];
person_len = size(person_subfolders, 1);

% Define out video path
%out_video_path = 'Datasets\June2019\Amplified_Blood_Flow_Without_Attenuation\';
% out_video_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Blood_Flow_After_Attenuation\Forehead\';
%out_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead_Avg_Images_RGB\';
out_path = 'C:\Users\konsg\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\DCT_cube_forehead\';
x=1;
l=1;
%dct_frame(1:71,1:101,29)=NaN;
%dct_output(1:71,1:101,29)=NaN;
%dct_out(1:71,1:101,29)=NaN;
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
        vpath = [video_path day_subfolders(i, :) person_subfolders(j, :)]
        files = dir(vpath);
       
        % For every video in vpath
        for k=3:length(files)
            
            % Load video
            vfilename = [vpath files(k).name];
            v = VideoReader(vfilename);
            nframes = v.duration * v.framerate
            
            n = 1; % how many avg frames for this video
            for s=1:31:nframes                
                
                % Extract frame
                %sum_frame = readFrame(v);
                
                for f=1:30
                    f
                    if hasFrame(v)
                        frame = readFrame(v);                    
                        dct_frame(:,:,x) = double(rgb2gray(frame));  % add new frame to sum
                        x=x+1;
                    else
                        break;
                    end
                end
       x=1;     
                
       for l=1:size(frame,1)
            for m=1:size(frame,2)%gia to kathe pixel sto xrono kano dct
                   dct_output(l,m,:)=dct(dct_frame(l,m,:));%kano ana 30 frame dct kai ta apothikeuo sto dct_output
                
            end
       end
       cube_max = max(dct_output(:));
       cube_min = min(dct_output(:));
       Range = cube_max - cube_min;
       dct_output = ((dct_output - cube_min)/Range);%Anrm = ((A - Amin)/Range - 0.5) * 2;  
       image_filename = [out_path day_subfolders(i, :) person_subfolders(j, :) 'day' num2str(i) '_video' num2str(k-2) '_dct_cube_' num2str(n) '.mat'];
       if(size(dct_output,3)==30)
       save(image_filename,'dct_output')
       else
           counter_n=counter_n+1;
       end
%                 for l=1:29                
%                 image_filename = [out_path day_subfolders(i, :) person_subfolders(j, :) 'day' num2str(i) '_video' num2str(k-2) '_avg_image_' num2str(n) '.jpg'];
%                 im=uint8(dct_output(:,:,l));
%                 imwrite(im, image_filename);
%                 
%                 
                 n = n + 1;
%                 end
            end            
        end
    end
end
            

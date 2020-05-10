clear all
close all
clc

% Step 2
% This code attenuates big movements that exist in the video in order for
% them not to be amplified to the next step 

% Define video paths
%video_path = 'Datasets\June2019\Cropped_Videos\';
%video_path = 'Datasets\June2019\Three_Videos_Out\Original\Forehead\';
video_path = 'C:\Users\konsg\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Original\Forehead\';
%subfolders = dir(video_path);
day_subfolders = ['Day1\'; 'Day2\'; 'Day3\'];
day_len = size(day_subfolders, 1);
person_subfolders = ['01\';'02\';'03\';'04\';'05\'];
person_len = size(person_subfolders, 1);

% Define out video path
%out_video_path = 'Datasets\June2019\Three_Videos_Out\Attenuated_Videos\Forehead\';
out_video_path = 'C:\Users\konsg\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Attenuated_Videos\Forehead\';

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
            
            % Construct video writer
            [fpath, name, ext] = fileparts(files(k).name);
            out_vfilename = [out_video_path day_subfolders(i, :) person_subfolders(j, :) name '_attenuated'] 
            
            tic
            
            motionAttenuateFixedPhase(vfilename, out_vfilename);
            
            toc
            
        end
    end
end
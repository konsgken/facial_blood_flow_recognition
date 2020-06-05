clear all
close all
clc

% Step1:
% This code detects the face (bounding box) at the 1st video frame and crops
% the frame in order to keep only the info needed (the face) using the
% coordinates of extracted bounding box
addpath(genpath(pwd));
% Set directory
cd 'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\AAM\face-release1.0-basic\';

% Compile library
disp('Compiling library. Please, wait...');
compile; 

addpath(genpath('C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\'));

% Load model for face detection
load face_p99.mat; % model

% Load AAM model
load cGN_DPM.mat;

% Set directory
cd 'C:\Users\konsg\OneDrive\facial_blood_flow_recognition\';

% Define video paths
video_path = 'Datasets\June2019\Original_Videos\';
%subfolders = dir(video_path);
day_subfolders = ['Day1\'; 'Day2\'; 'Day3\'];
day_len = size(day_subfolders, 1);
person_subfolders = ['01\';'02\';'03\';'04\';'05\'];
person_len = size(person_subfolders, 1);

% Define out video path
out_video_path = 'Datasets\June2019\Three_Videos_Out\Original\';

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
            
            % Construct video writer
            [fpath, name, ext] = fileparts(files(k).name);
            
            % Forehead
            out_vfilename = [out_video_path 'Forehead\' day_subfolders(i, :) person_subfolders(j, :) name] 
            vwriter = VideoWriter(out_vfilename, 'Uncompressed AVI');
            open(vwriter);
            
            tic
            
            % Extract 1st frame
            frame = readFrame(v);

            % rect: bbox of detected face
            [fitbox, rect] = detect_face(frame, model, 512, 512);  % 512 x 512 output frame
            
            % Fitting AAM
            iter = 5;
            fitted_shape=GN_DPM_fit(frame, fitbox, cGN_DPM, iter);      
            
            % Find forehead
            
            % Sort fitted_shape / column
            fitted_shape_x_sorted = sort(fitted_shape(:, 1), 'ascend');
            fitted_shape_y_sorted = sort(fitted_shape(:, 2), 'ascend');
            
            min_x = fitted_shape_x_sorted(1);
            max_x = fitted_shape_x_sorted(length(fitted_shape_x_sorted));
            
            % Find center of face
            center = (max_x - min_x)/2;
            threshold = min_x + center;
            
            % Find eye brows highest points
            [p1_x p1_y] = find(fitted_shape(:, 2) == fitted_shape_y_sorted(1));    
            
%             k = 2;         
%             [p2_x p2_y] = find(fitted_shape(:, 2) == fitted_shape_y_sorted(k));
%             
%              while(fitted_shape(p2_x, 1) <= threshold)
%                  k = k + 1;
%                  [p2_x p2_y] = find(fitted_shape(:, 2) == fitted_shape_y_sorted(k));
%              end
%                                    
            % Find forehead
            forehead_h = 70;
            forehead_w = 201;
            
            % Create forehead rectangle
            forehead_rect = [fitted_shape_x_sorted(1), fitted_shape(p1_x, 2)-forehead_h, forehead_w, forehead_h];
            
            % For every frame
            while hasFrame(v)
                
                % Cropped face 
                cropped_frame = imcrop(frame, forehead_rect);
                
                % Write cropped frame in new video
                writeVideo(vwriter, cropped_frame);
                
                % Read new frame
                frame = readFrame(v);
            end
            
            close(vwriter);
            toc
            
        end
    end
end
    



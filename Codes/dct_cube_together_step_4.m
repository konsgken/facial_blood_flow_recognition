clear all
close all
clc

% This code creates the avg image (1 image/30 frames), so we have 1 avg
% image/ sec

% Define video paths
%video_path = 'Datasets\June2019\Cropped_Videos\';
%video_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead\';
image1_path = 'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\DCT_cube_forehead\';
image2_path = 'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\DCT_cube_left_cheeck\';
image3_path = 'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\DCT_cube_right_cheeck\';
%subfolders = dir(video_path);
day_subfolders = ['Day1\'; 'Day2\'; 'Day3\'];
day_len = size(day_subfolders, 1);
person_subfolders = ['01\';'02\';'03\';'04\';'05\';'06\';'07\';'08\';'09\';'10\';'11\';'12\'];
person_len = size(person_subfolders, 1);

% Define out video path
%out_video_path = 'Datasets\June2019\Amplified_Blood_Flow_Without_Attenuation\';
% out_video_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Blood_Flow_After_Attenuation\Forehead\';
%out_path = 'Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\Forehead_Avg_Images_RGB\';
out_path = 'C:\Democritus University of Thrace (DUTh)\OneDrive\facial_blood_flow_recognition\Datasets\June2019\Three_Videos_Out\Amplified_Only_Blood_Flow_After_Attenuation\DCT_cube_together\';

alpha = 120;
level = 4;
fl = 0.83;
fh = 1;
samplingRate = 30;
chromAttenuation = 1;
n=1;
% For every day 
for i=1:day_len
    
    % For every person
    for j=1:person_len
        
        % Define video path
        ipath1 = [image1_path day_subfolders(i, :) person_subfolders(j, :)];
        ipath2 = [image2_path day_subfolders(i, :) person_subfolders(j, :)];
        ipath3 = [image3_path day_subfolders(i, :) person_subfolders(j, :)];
        files1 = dir(ipath1);
        files2 = dir(ipath2);
        files3 = dir(ipath3);
       
        % For every video in vpath
        for k=3:length(files1)
            
            % Load video
            ifilename1 = [ipath1 files1(k).name];
            ifilename2 = [ipath2 files2(k).name];
            ifilename3 = [ipath3 files3(k).name];
            cube_1 = load(ifilename1);
            cube_2 = load(ifilename2);
            cube_3 = load(ifilename3);
            cube_forehead = cube_1.dct_output;
            cube_left_cheeck = cube_2.dct_output;
            cube_right_cheeck = cube_3.dct_output;
%             cubes_together(1:size(cube_right_cheeck,1),1:size(cube_right_cheeck,2),:)=cube_right_cheeck;
%             cubes_together(size(cube_right_cheeck,1)+1:size(cube_right_cheeck,1)+size(cube_left_cheeck,1),1:size(cube_left_cheeck,2),:)=cube_left_cheeck;
%             cubes_together(size(cube_right_cheeck,1)+1:size(cube_right_cheeck,1)+size(cube_left_cheeck,1),size(cube_left_cheeck,2)+1:size(cube_left_cheeck,2)+size(cube_left_cheeck,2),:)=cube_forehead;
            cubes_left_right=cat(2,cube_left_cheeck,cube_right_cheeck);
            cubes_together=cat(1,cube_forehead,cubes_left_right);
            out_filename = [out_path day_subfolders(i, :) person_subfolders(j, :) 'day' num2str(i) '_video' num2str(k-2) '_cube_together_' num2str(n) '.mat']
            
            save(out_filename,'cubes_together');            
            n=n+1;
            
                   
        end
    end
end
            

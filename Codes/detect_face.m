function [fitbox, rect] = detect_face(image, model, fheight, fwidth)

% SUMMARY:
% This function returns:
% 1. the fitbox around the detected face to use for the AAM
% 2. the rectangle around the detected face to use for crop purposes
% image : the image where we want to make the detection
% model : the model used for face detection
% fheight, fwidth: if we want to have a fitbox of fixed dimensions (fheight x
% fwidth)
  
     
    % Detect the face
    bs = detect(image, model, model.thresh);
    bs = clipboxes(image, bs);
    bs = nms_face(bs, 0.3);
    bs = bs(1);
    
    % Face detected, extracting bbox
	points = [(bs.xy(:,1)+bs.xy(:,3))/2 , (bs.xy(:,2)+bs.xy(:,4))/2];
	x_min = round(min(bs.xy(:,1)));
	x_max = round(max(bs.xy(:,3)));
	x_avg = round((x_min+x_max)/2);
	y_min = round(min(bs.xy(:,2)));
	y_max = round(max(bs.xy(:,4)));
    
    % Change size
%     extra_height = (fheight + y_min - y_max)/2;
% 	extra_width = (fwidth + x_min - x_max)/2;
%     
%     up = 50; % to get more forehead
%     
%     x_min = x_min - abs(extra_width);
%     x_max = x_max + abs(extra_width);
%     y_min = y_min - abs(extra_height) - up;
%     y_max = y_max + abs(extra_height) -up;      
    
    % Fitbox
    fitbox=[y_min,x_min,y_max-y_min,x_max-x_min]; %bounding box for AAM is reverse
    
    % Rectangle  
	rect = [x_min, y_min, x_max-x_min, y_max-y_min]; %bounding box for AAM is reverse  
    
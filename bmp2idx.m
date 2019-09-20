function [net, info] = bmp2idx()
%CNN_MNIST  Demonstrates MatConvNet on MNIST
% This function takes all the bitmap images in the training and test sets
% and forms two idx files housing all the images and two idx files
% labelling them as index gestures or a negative.

% Starts a timer on the process for personal use
tic;

% Sets dimensions of the picture set, in this case, 28x28
imlen=28;
imwid=imlen;
% Searches for the folder 'K:/YxY/Training Set/' where Y is the size of the
% image, in this case it's 28 but can be adjusted for larger sizes
folder = strcat('K:/', int2str(imwid), 'x',...
    int2str(imlen), '/Training Set/');
% Fullfile pulls all the bitmap images in the folder defined above, dir
% lists them in order
files = dir(fullfile(folder, '*.bmp'));

% Creates file which these images will be stored in, both in the idx format
fileIDdata = fopen(strcat('training_data_', int2str(imwid),...
    'x', int2str(imlen), '.idx'), 'w');
fileIDlabel = fopen(strcat('training_labels_', int2str(imwid),...
    'x', int2str(imlen), '.idx'), 'w');

% writes files in big endian writing, which is how memory is stored. First
% set relates to training images where the preamble value is defined as
% 2501 then the size is defined along with the dimensions
% uint32 is the integer type and ieee-be is how it's read
fwrite(fileIDdata, 2051, 'uint32', 'ieee-be');
fwrite(fileIDdata, 150000, 'uint32', 'ieee-be');
fwrite(fileIDdata, imlen, 'uint32', 'ieee-be');
fwrite(fileIDdata, imwid, 'uint32', 'ieee-be');

% Variation for the training labels. Dimensions are not needed, 2049 used
fwrite(fileIDlabel, 2049, 'uint32', 'ieee-be');
fwrite(fileIDlabel, 150000, 'uint32', 'ieee-be');

% Counter started for iterations for personal use
z=0;
% Reads every file in directory started in line 15
for file = files'
    z=z+1;
    disp(z);
    % Concatenates folder name with individual image for full directory
    f = strcat(folder, file.name);
    % Reads image as a bitmap
    RGB = imread(f, 'bmp');
    % Grayscales image which makes calculations more expedient although
    % full colour tests could also be considered
    I = rgb2gray(RGB);
    disp(f)
    % Writes the result to the image idx file, to be stored for later use
    fwrite(fileIDdata, I, 'uint8', 'ieee-be');
    % Determines whether image is labelled as an index finger. If it is 
    % then marked 3, 1 otherwise (binary notation, 0, 1 confuses matlab)
    if (findstr(file.name, strcat('K:/', int2str(imwid), 'x',...
            int2str(imlen), '/Training Set/Index')))
        fwrite(fileIDlabel, 3, 'uint8', 'ieee-be');
    else 
        fwrite(fileIDlabel, 1, 'uint8', 'ieee-be');
    end
end

% Closes idx files now that they are complete
fclose(fileIDdata);
fclose(fileIDlabel);
% Same procedure repeated for Test Set
folder = 'K:/28x28/Test Set/';
files = dir(fullfile(folder, '*.bmp'));

fileIDdata = fopen(strcat('test_data_', int2str(imwid), 'x',...
    int2str(imlen), '.idx'), 'w');
fileIDlabel = fopen(strcat('test_labels_', int2str(imwid), 'x',...
    int2str(imlen), '.idx'), 'w');

% big endian writing
fwrite(fileIDdata, 2051, 'uint32', 'ieee-be');
fwrite(fileIDdata, 50000, 'uint32', 'ieee-be');
fwrite(fileIDdata, imlen, 'uint32', 'ieee-be');
fwrite(fileIDdata, imwid, 'uint32', 'ieee-be');

fwrite(fileIDlabel, 2049, 'uint32', 'ieee-be');
fwrite(fileIDlabel, 50000, 'uint32', 'ieee-be');

z = 0;
for file = files'
    z=z+1;
    disp(z);
        
    f = strcat(folder, file.name);
    RGB = imread(f, 'bmp');
    I = rgb2gray(RGB);
    
    disp(f)
    fwrite(fileIDdata, I, 'uint8', 'ieee-be');

    if (findstr(file.name, strcat('K:/', int2str(imwid), 'x',...
            int2str(imlen), '/Test Set/Index')))
        fwrite(fileIDlabel, 3, 'uint8', 'ieee-be');
    else 
        fwrite(fileIDlabel, 1, 'uint8', 'ieee-be');
    end
end
% closes idx files now that they are complete
fclose(fileIDdata);
fclose(fileIDlabel);
% Ends function, outputs time taken (200000 took about 30 mins on my
% computer but would be quicker on NoMachine)
toc;
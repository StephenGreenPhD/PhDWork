% format long is used for this as putting 200000 images through this code
% means that it helps to have as much extra precision as possible even if
% the total effect could be negligable.
format long
% Much like the setup function, this retrieves all data in the MasterIndex
% folder. This code was run twice, one on a small set of 100 images, split
% 50/50 between true and false readings, it was also put through 200000
% images with a split of 20000 indexes and 180000 negatives which is what
% this code has been specifically designed to do.
folder = strcat('K:\MasterIndex\');
files = dir(fullfile(folder, '*.bmp'));
z=0;
a=0;
b=0;
BestScoreIndex=0;
BestScoreNegative=0;
A=zeros(28,28);
B=zeros(28,28);
Index=zeros(1,20000);
Negative=zeros(1,180000);
% This uses the same reading process as bmp2idx to parse through so many
% images at once however in this case I am taking a reading from each image
% rather than transferring them to a secure container.
for file = files'
    z=z+1;
    disp(z);
    f = strcat(folder, file.name);
    RGB = imread(f, 'bmp');
    I = rgb2gray(RGB);
    % At this point the uint8 mode present in the grayscale version of the
    % original image had to be converted to double. This was because I
    % needed the specific values to be added to each other and uint8 has a
    % hard cap of 255 so running through the program would produce a black 
    % square so double precision is the best choice.
    J=double(I);
    IndexRecognition(f);
    % A search is conducted for Index images (the files were named Index
    % (1), Index (2) etc). If there was a match first of all that picture's
    % score would be added to a running tally then that images pixel values
    % would be added to the square. At the end would be a 28x28 square with
    % values between 0 and 255 which can then be read back as an image. The
    % scores were also kept in a chart so they could be analysed further
    % for normalisation
    if (findstr(file.name, 'Index'))
        BestScoreIndex=BestScoreIndex+ans;
        A=A+(J/20000);
        a=a+1;
        Index(1,a)=ans;
    else
        BestScoreNegative=BestScoreNegative+ans;
        B=B+(J/180000);
        b=b+1;
        Negative(1,b)=ans;
    end
end
% The scores were divided by their size to produce the average value
% for each sector. These can be assessed to show how well the system works.
BestScoreIndex=BestScoreIndex/20000;
BestScoreNegative=BestScoreNegative/180000;
% The overlapping images are then outputted with the average score on top.
imshow(A)
title(BestScoreIndex);
imshow(B)
title(BestScoreNegative);
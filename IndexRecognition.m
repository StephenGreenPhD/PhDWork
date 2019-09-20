function[bestScore] = IndexRecognition(x)
% This function inputs an image and produces a score that shows how well
% the image matches with what exists in the training and test sets.

% First MatConvNet is setup by running the setup neural network that was
% used in the previous code that defines the convolutional neural network
% that holds up the transformations applied in the setup phase.
run ../../matlab/vl_setupnn ;

% This loads the model created in the setup code, net_stephen.mat is the
% main output of the previous step and is what's used to compare against
% the input of this function.
net = load('net_stephen.mat') ;
% The net in this case is the convolutional neural network. This step
% retrieves it from the MatConvNet architecture
net = vl_simplenn_tidy(net.net) ;
% Here there's a problem as an error occurs normally as when iterating for
% each digit between 0-9 in the regular case, the final layer softmaxloss
% resets to softmax so more calculations can occur. Since this has binary
% inputs the initial reset never happens, so I've added a line that does
% this manually so there is no issue running the CNN afterwards.
net.layers{1,8}.type='softmax';
% The input image x is read by the code and stored as the variable 'im'
im = imread(x) ;
% The image is automatically resized to 28x28 if necessary
im = imresize(im,[28,28]);
% It is then grayscaled to put it in line with the training/test set data
im = rgb2gray(im);
% The image is then converted to single precision so it can go through the
% CNN to attain a score reading
im = im2single(im);
res = vl_simplenn(net, im) ;
% The final score is then attained by drawing from all the results attained
% in the network with the gather function, deleting all single columns with 
% the squeeze function and outputting a given score.
scores = squeeze(gather(res(end).x)) ;
% the best variable determines what the likely reading is which is useless
% in this case since the value is binary however bestScore shows how well
% it matches the positive data compared to the negative and is the final
% output of this recognition function.
[bestScore, best] = max(scores) ;
end
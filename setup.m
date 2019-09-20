function [net, info] = setup(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

% The next six functions were kept mostly intact from their MNIST
% counterparts. Of note is that it runs the setup neural network first and
% then builds off it, variables 1:10 were changed to 1:2 since my program
% would be working with boolean variables only. All respective data would
% be moved to the file stephens-baseline- for further use. At the end the
% code will run 20 epochs of 100 images training the network to
% successfully recognise index gestures.
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['stephens-baseline-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'stephens') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
                     'networkType', opts.networkType) ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:2,...
    'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

disp(net.meta.classes.name);
save('net_stephen.mat','net')

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Prepare the imdb structure, returns image data with mean image subtracted
% The main change in the MNIST code occurs here where the directories are
% changed (set to 'r' to symbolise read only files), the vectors are then
% reshaped accordingly.
f=fopen('C:\Users\grims_000\Documents\MATLAB\matconvnet-master\matconvnet-master\work\phd\training_data_28x28.idx','r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
% The first four bytes hold the magic number determining that this is a
% training set, the next four hold the size (150000 in this case), and the
% next two sets of four hold the length and width respectively which in
% this case is 28. The rest are then the values reshaped into one long
% vector.
x1=permute(reshape(x1(17:end),28,28,150000),[2 1 3]) ;

f=fopen('C:\Users\grims_000\Documents\MATLAB\matconvnet-master\matconvnet-master\work\phd\test_data_28x28.idx','r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,50000),[2 1 3]) ;

f=fopen('C:\Users\grims_000\Documents\MATLAB\matconvnet-master\matconvnet-master\work\phd\training_labels_28x28.idx','r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
% For the labels since the size is unnecessary there are only eight bytes
% needed to hold the number 2049 and the size of the label set.
y1=double(y1(9:end)')+1 ;

f=fopen('C:\Users\grims_000\Documents\MATLAB\matconvnet-master\matconvnet-master\work\phd\test_labels_28x28.idx','r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;
% These next ten lines reshape all the terms that have been preestablished
% so further calculations can happen.
set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:2,'uniformoutput',false) ;

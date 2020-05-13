addpath('../packages/tensorlab_2016-03-28');
addpath('../packages/tensor_toolbox-v3.1');
addpath('../packages/onlineCP');
warning('off', 'all');

% OnlineCP w. trigger in full video
% R=170;OnlineCP_trigger

currentPath = fileparts(mfilename('fullpath'));
options.Display = false; % Show progress on the command line.
options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
options.Refinement = false;
threshold = 1.5;

startFrame = 0;
endFrame = 205;

numOfFrames = endFrame - startFrame;
tao = 5;
% 205 * 180(31~210) * 320 * 3
dims = [180 320 3 205];
iterFrame = 5;
videoTensor = NaN(dims);
frameRate = 30 * 0.3;
minibatchSize = 1;

outputVideo = VideoWriter('OPT/title');
outputVideo.FrameRate = frameRate;
open(outputVideo);

for frame = 1:numOfFrames
    img = ConvertText2Image(sprintf('#%03d', frame+startFrame));
    img = imresize(img, [180, 320], 'nearest');
    % disp(sprintf('#%03d', frame))
    % imshow(img)
    writeVideo(outputVideo, img);
end
close(outputVideo);


outputVideo = VideoWriter('OPT/org');
outputVideo.FrameRate = frameRate;
open(outputVideo);

for i = startFrame/iterFrame:endFrame/iterFrame-1
    tensorFile = fopen(strcat(currentPath, '/data/video', num2str(i), '.tensor'), 'r');
    tic;
    X = fscanf(tensorFile, "%d %d %d %d %d", [5, inf]);
    for row = X
        if row(2) > 30 & row(2) <= 210
            videoTensor(row(2)-30, row(3), row(4), row(1)-startFrame) = row(5);
        end
    end
    fclose(tensorFile);
    toc;
end

for frame = 1:numOfFrames
    img = uint8(squeeze(videoTensor(:, :, :, frame)));
    writeVideo(outputVideo,img);
end
close(outputVideo);

outputVideo = VideoWriter(strcat('OPT/trigger-',num2str(R)));
outputVideo.FrameRate = frameRate;
open(outputVideo);

for frame = 1:tao
    img = uint8(squeeze(videoTensor(:, :, :, frame)));
    writeVideo(outputVideo,img);
end


% tic;
idx = repmat({':'}, 1, length(dims));
idx(end) = {1:tao};
initX = videoTensor(:, :, :, 1:tao);

initAs = cpd(initX, R, options);

[onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
onlineAs = initAs(1:end-1);
onlineAs_N = initAs{end};

Uest = [onlineAs'; {onlineAs_N}];
Test = cpdgen(Uest);
whos Test initX
prevImgErr = frob(Test(:,:,:,end)-initX(:,:,:,end))

toc;
for t = 1:minibatchSize:numOfFrames-tao
    frame = tao+t;
    fprintf('\n> %dth frame\n', frame+startFrame);
    endTime = min(tao+t+minibatchSize-1, numOfFrames);
    idx(end) = {tao+t:endTime};
    
    x = squeeze(videoTensor(idx{:}));
    idx(end) = {1:endTime};
    Xt = videoTensor(idx{:});
    tic;
    [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
    onlineAs_N(end+1,:) = onlineAlpha;
    Uest = [onlineAs'; {onlineAs_N}];

    As1 = Uest{1};
    As2 = Uest{2};
    As3 = Uest{3};
    As4 = Uest{4};

    Test = cpdgen(Uest);
    imgEst = squeeze(Test(:, :, :, frame));
    imgOrg = squeeze(Xt(:, :, :, frame));
    testImgErr1(t) = frob(imgEst-imgOrg);


    if prevImgErr * threshold < testImgErr1(t)
        disp('Drastic scene detected. CP-ALS update triggered!');
        initAs = cpd(Xt, R, options);
    
        [onlinePs, onlineQs] = onlineCP_initial_tenlab(Xt, initAs, R);
        onlineAs = initAs(1:end-1);
        onlineAs_N = initAs{end};
    
        Uest = [onlineAs'; {onlineAs_N}];
        Test = cpdgen(Uest);
        imgEst = squeeze(Test(:, :, :, frame));
        whos initAs onlineAs onlineAs_N
    end
    toc;

    testFrame(t) = frame+startFrame;
    testRuntime(t) = toc;
    testNormErr(t) = frob(Test-Xt);
    testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;
    testImgErr(t) = frob(imgEst-imgOrg);
    prevImgErr = testImgErr(t);
    writeVideo(outputVideo,uint8(imgEst));
end
whos As1 As2 As3 As4

close(outputVideo);
fileID = fopen(strcat('OPT/trigger-',num2str(R),'.txt'),'w');
testRuntime_Fitness = [testFrame', testRuntime', testFitness', testImgErr', testImgErr1'];
testRuntime_Fitness = testRuntime_Fitness(1:numOfFrames-tao, :);
result = sprintf('%d\t%.4fs\t%.4f%%\t%.f\t%.f\n', testRuntime_Fitness')
fprintf(fileID, '%s', result);
fclose(fileID);
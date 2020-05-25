addpath('../packages/tensorlab_2016-03-28');
addpath('../packages/tensor_toolbox-v3.1');
addpath('../packages/onlineCP');
warning('off', 'all');

% OnlineCP w. split in full video
% R=10;OnlineCP_split

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
% startFrame = 0;
% endFrame = 205;

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

outputVideo = VideoWriter(strcat('OPT/split-',num2str(R)));
outputVideo.FrameRate = frameRate;
open(outputVideo);

idx = repmat({':'}, 1, length(dims));
prevDrasticFrame = 1;

for frame = 1:minibatchSize:numOfFrames
    t = frame;
    fprintf('\n> %dth frame\n', frame+startFrame);
    endTime = min(frame+minibatchSize-1, numOfFrames);
    idx(end) = {frame:endTime};
    x = squeeze(videoTensor(idx{:}));

    idx(end) = {prevDrasticFrame:endTime};
    Xt = videoTensor(idx{:});
    imgOrg = squeeze(Xt(:, :, :, frame-prevDrasticFrame+1));

    tic;

    if frame - prevDrasticFrame < tao-1
        disp('continue.');
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        continue;

    elseif frame-prevDrasticFrame+1 == tao
        disp('OnlineCP init! CP-ALS update!');
        initAs = cpd(Xt, R, options);
    
        [onlinePs, onlineQs] = onlineCP_initial_tenlab(Xt, initAs, R);
        onlineAs = initAs(1:end-1);
        onlineAs_N = initAs{end};
    
        Uest = [onlineAs'; {onlineAs_N}];
        Test = cpdgen(Uest);

        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        for i = 1:tao
            prevFrame = frame-prevDrasticFrame+1-tao+i
            imgEst = squeeze(Test(:, :, :, prevFrame));
            imgOrg = squeeze(Xt(:, :, :, prevFrame));
            if i > 1
                testImgNormErr1(t-tao+i) = frob(imgEst-imgOrg);
            end
            testImgNormErr(t-tao+i) = frob(imgEst-imgOrg);
            writeVideo(outputVideo,uint8(imgEst));
        end
        prevImgNormErr = testImgNormErr(t);
        continue;

    else
        disp('OnlineCP update!');
        [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
        onlineAs_N(end+1,:) = onlineAlpha;
        Uest = [onlineAs'; {onlineAs_N}];
    
        As1 = Uest{1};
        As2 = Uest{2};
        As3 = Uest{3};
        As4 = Uest{4};
    
        Test = cpdgen(Uest);
        imgEst = squeeze(Test(:, :, :, frame-prevDrasticFrame+1));
        testImgNormErr1(t) = frob(imgEst-imgOrg);
    
        % print log
        [prevImgNormErr, testImgNormErr1(t)]
    
        if prevImgNormErr*threshold < testImgNormErr1(t)
            disp('Drastic scene detected. CP-ALS update triggered!');
            prevDrasticFrame = frame;
            testFrame(t) = frame+startFrame;
            testRuntime(t) = toc;
            continue;
        end
    end
    toc;

    testFrame(t) = frame+startFrame;
    testRuntime(t) = toc;
    testImgNormErr(t) = frob(imgEst-imgOrg);
    prevImgNormErr = testImgNormErr(t);
    writeVideo(outputVideo,uint8(imgEst));
end
whos As1 As2 As3 As4

close(outputVideo);
fileID = fopen(strcat('OPT/split-',num2str(R),'.txt'),'w');
testRuntime_Fitness = [testFrame', testRuntime', testImgNormErr', testImgNormErr1'];
result = sprintf('%d\t%.4f\t%.f\t%.f\n', testRuntime_Fitness')
fprintf(fileID, '%s', result);
fclose(fileID);

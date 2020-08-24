addpath('../tensorlab_2016-03-28');
addpath('../tensor_toolbox-v3.1');
addpath('../onlineCP');
warning('off', 'all');

% 205 * 180(31~210) * 320 * 3
% >> opt=1;OnlineCP_OPT
% >> opt=2;OnlineCP_OPT
% >> opt=3;OnlineCP_OPT
% >> opt=4;OnlineCP_OPT
% >> opt=5;OnlineCP_OPT
% >> opt=6;OnlineCP_OPT
currentPath = fileparts(mfilename('fullpath'));
R = 10;
% dims = [205 180 320 3];
% numOfFrames = dims(1);
% tao = 100;

startFrame = 115
endFrame = 140;

numOfFrames = endFrame - startFrame;
tao = 5;
dims = [180 320 3 205];
iterFrame = 5;
N = numOfFrames / iterFrame;
videoTensor = NaN(dims);
frameRate = 30 * 0.1;
minibatchSize = 1;

outputVideo = VideoWriter('OPT/org');
outputVideo.FrameRate = frameRate;
open(outputVideo);

% for i = 1:N
for i = startFrame/iterFrame+1:endFrame/iterFrame
    tensorFile = fopen(strcat(currentPath, '/video', num2str(i-1), '.tensor'), 'r');
    tic;
    X = fscanf(tensorFile, "%d %d %d %d %d", [5, inf]);
    for row = X
        if row(2) > 30 & row(2) <= 210
            videoTensor(row(2)-30, row(3), row(4), row(1)-startFrame) = row(5);
            % videoTensor(row(2)-30, row(3), row(4), row(1)) = row(5);
        end
    end
    fclose(tensorFile);
    toc;
end

for frame = 1:numOfFrames
    if frame == tao
        img = imread(strcat('OPT/opt.png'));
        writeVideo(outputVideo,img);
        writeVideo(outputVideo,img);
        writeVideo(outputVideo,img);
        writeVideo(outputVideo,img);
    end
    img = uint8(squeeze(videoTensor(:, :, :, frame)));
    writeVideo(outputVideo,img);
end
close(outputVideo);

outputVideo = VideoWriter(strcat('OPT/opt', num2str(opt)));
outputVideo.FrameRate = frameRate;
open(outputVideo);

for frame = 1:tao
    img = uint8(squeeze(videoTensor(:, :, :, frame)));
    writeVideo(outputVideo,img);
end


img = imread(strcat('OPT/opt', num2str(opt), '.png'));
img = imresize(img, [180 320]);
imshow(img)
writeVideo(outputVideo,img);
writeVideo(outputVideo,img);
writeVideo(outputVideo,img);
writeVideo(outputVideo,img);


if opt == 1
    % tic;
    idx = repmat({':'}, 1, length(dims));
    idx(end) = {1:tao};
    initX = videoTensor(:, :, :, 1:tao);

    options.Display = false; % Show progress on the command line.
    options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
    options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
    options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
    options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
    options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
    options.Refinement = false;
    initAs = cpd(initX, R, options);

    [onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
    onlineAs = initAs(1:end-1);
    onlineAs_N = initAs{end};
    whos onlineAs_N
    toc;

    for t = 1:minibatchSize:numOfFrames-tao
        fprintf('the %dth steps\n', t);
        % get the incoming slice
        frame = tao+t;
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
        % [As1(1:3,1), As2(1:3,1), As3(1:3,1)]

        Test = cpdgen(Uest);
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        testNormErr(t) = frob(Test-Xt);
        testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;
        toc;
        Test = permute(cpdgen(Uest), [4 1 2 3]);
        img = uint8(squeeze(Test(frame, :, :, :)));
        writeVideo(outputVideo,img);
    end
    whos As1 As2 As3 As4
elseif opt == 2
    for t = 1:minibatchSize:numOfFrames-tao
        frame = tao+t;
        disp(frame);
        tic;
        T = videoTensor(:, :, :, 1:frame);
        whos T
        options.Display = true; % Show progress on the command line.
        options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
        options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
        options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
        options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
        options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
        options.Refinement = false;
        options.incomplete = false;
        Uest = cpd(T, R, options);
        % whos Uest
        Test = cpdgen(Uest);

        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        testNormErr(t) = frob(Test-T);
        testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;

        toc;
        Test = permute(cpdgen(Uest), [4 1 2 3]);
        img = uint8(squeeze(Test(frame, :, :, :)));
        writeVideo(outputVideo, img);
    end
elseif opt == 3
    % tic;
    newTao = drasticFrame - startFrame;
    idx = repmat({':'}, 1, length(dims));
    idx(end) = {1:newTao};
    initX = videoTensor(:, :, :, 1:newTao);

    tic;
    options.Display = false; % Show progress on the command line.
    options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
    options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
    options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
    options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
    options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
    options.Refinement = false;
    initAs = cpd(initX, R, options);

    [onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
    onlineAs = initAs(1:end-1);
    onlineAs_N = initAs{end};


    Uest = [onlineAs'; {onlineAs_N}];
    Test = permute(cpdgen(Uest), [4 1 2 3]);
    whos onlineAs_N

    for t = 1:newTao-tao
        frame = tao+t;
        idx(end) = {1:frame};
        Xt = permute(videoTensor(idx{:}), [4 1 2 3]);
        whos Xt Test
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        testNormErr(t) = frob(Test(1:frame,:,:,:)-Xt);
        testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;
        
        img = uint8(squeeze(Test(frame, :, :, :)));
        writeVideo(outputVideo, img);
    end

    for t = 1:minibatchSize:numOfFrames-newTao
        fprintf('the %dth steps\n', t);
        % get the incoming slice
        frame = newTao+t;
        endTime = min(newTao+t+minibatchSize-1, numOfFrames);
        idx(end) = {newTao+t:endTime};
        
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
        testFrame(t+newTao-tao) = frame+startFrame;
        testRuntime(t+newTao-tao) = toc;
        testNormErr(t+newTao-tao) = frob(Test-Xt);
        testFitness(t+newTao-tao) = (1-testNormErr(t)/frob(Xt))*100;
        toc;
        Test = permute(cpdgen(Uest), [4 1 2 3]);
        img = uint8(squeeze(Test(frame, :, :, :)));
        writeVideo(outputVideo,img);
    end
elseif opt == 4
    % tic;
    newTao = drasticFrame - startFrame-1;
    idx = repmat({':'}, 1, length(dims));
    idx(end) = {1:newTao};
    initX = videoTensor(:, :, :, 1:newTao);

    tic;
    options.Display = false; % Show progress on the command line.
    options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
    options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
    options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
    options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
    options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
    options.Refinement = false;
    initAs = cpd(initX, R, options);

    [onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
    onlineAs = initAs(1:end-1);
    onlineAs_N = initAs{end};


    Uest = [onlineAs'; {onlineAs_N}];
    Test = permute(cpdgen(Uest), [4 1 2 3]);
    whos onlineAs_N

    for t = 1:newTao-tao
        frame = tao+t;
        idx(end) = {1:frame};
        Xt = permute(videoTensor(idx{:}), [4 1 2 3]);
        whos Xt Test
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        testNormErr(t) = frob(Test(1:frame,:,:,:)-Xt);
        testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;
        
        img = uint8(squeeze(Test(frame, :, :, :)));
        writeVideo(outputVideo, img);
    end

    for t = 1:minibatchSize:numOfFrames-newTao
        fprintf('the %dth steps\n', t);
        % get the incoming slice
        frame = newTao+t;
        endTime = min(newTao+t+minibatchSize-1, numOfFrames);
        idx(end) = {newTao+t:endTime};
        
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
        testFrame(t+newTao-tao) = frame+startFrame;
        testRuntime(t+newTao-tao) = toc;
        testNormErr(t+newTao-tao) = frob(Test-Xt);
        testFitness(t+newTao-tao) = (1-testNormErr(t)/frob(Xt))*100;
        toc;
        Test = permute(cpdgen(Uest), [4 1 2 3]);
        img = uint8(squeeze(Test(frame, :, :, :)));
        writeVideo(outputVideo,img);
    end
elseif opt == 5
    % tic;
    threshold = 1.5;
    idx = repmat({':'}, 1, length(dims));
    idx(end) = {1:tao};
    initX = videoTensor(:, :, :, 1:tao);

    options.Display = false; % Show progress on the command line.
    options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
    options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
    options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
    options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
    options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
    options.Refinement = false;
    initAs = cpd(initX, R, options);

    imgEst = squeeze(Test(frame, :, :, :));
    imgOrg = squeeze(Xt(:, :, :, frame));

    [onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
    onlineAs = initAs(1:end-1);
    onlineAs_N = initAs{end};

    Uest = [onlineAs'; {onlineAs_N}];
    Test = cpdgen(Uest);
    whos Test initX
    prevImgErr = frob(Test(:,:,:,end)-initX(:,:,:,end))

    toc;
    for t = 1:minibatchSize:numOfFrames-tao
        fprintf('\n> %dth step\n', t);
        % get the incoming slice
        frame = tao+t;
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
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        testNormErr(t) = frob(Test-Xt);
        testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;
    

        Test = permute(cpdgen(Uest), [4 1 2 3]);
        imgEst = squeeze(Test(frame, :, :, :));
        imgOrg = squeeze(Xt(:, :, :, frame));
        testImgErr1(t) = frob(imgEst-imgOrg);


        if prevImgErr * threshold < testImgErr1(t)
            disp('Drastic scene detected. CP-ALS update triggered!');
            initAs = cpd(Xt, R, options);
        
            [onlinePs, onlineQs] = onlineCP_initial_tenlab(Xt, initAs, R);
            onlineAs = initAs(1:end-1);
            onlineAs_N = initAs{end};
        
            Uest = [onlineAs'; {onlineAs_N}];
            Test = permute(cpdgen(Uest), [4 1 2 3]);
            imgEst = squeeze(Test(frame, :, :, :));
            whos initAs onlineAs onlineAs_N
        end
        toc;

        testImgErr(t) = frob(imgEst-imgOrg);
        prevImgErr = testImgErr(t);
        writeVideo(outputVideo,uint8(imgEst));
    end
    whos As1 As2 As3 As4






elseif opt == 6
    % tic;
    threshold = 10;
    idx = repmat({':'}, 1, length(dims));
    idx(end) = {1:tao};
    initX = videoTensor(:, :, :, 1:tao);

    options.Display = false; % Show progress on the command line.
    options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
    options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
    options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
    options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
    options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
    options.Refinement = false;
    initAs = cpd(initX, R, options);

    imgEst = squeeze(Test(frame, :, :, :));
    imgOrg = squeeze(Xt(:, :, :, frame));

    [onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
    onlineAs = initAs(1:end-1);
    onlineAs_N = initAs{end};

    Uest = [onlineAs'; {onlineAs_N}];
    Test = cpdgen(Uest);
    whos Test initX
    prevImgErr = frob(Test(:,:,:,end)-initX(:,:,:,end))

    toc;
    for t = 1:minibatchSize:numOfFrames-tao
        fprintf('\n> %dth step\n', t);
        % get the incoming slice
        frame = tao+t;
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
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        testNormErr(t) = frob(Test-Xt);
        testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;
    

        Test = permute(cpdgen(Uest), [4 1 2 3]);
        imgEst = squeeze(Test(frame, :, :, :));
        imgOrg = squeeze(Xt(:, :, :, frame));
        testImgErr1(t) = frob(imgEst-imgOrg);


        if prevImgErr * threshold < testImgErr1(t)
            disp('Drastic scene detected. CP-ALS update triggered!');
            initAs = cpd(Xt, R, options);
        
            [onlinePs, onlineQs] = onlineCP_initial_tenlab(Xt, initAs, R);
            onlineAs = initAs(1:end-1);
            onlineAs_N = initAs{end};
        
            Uest = [onlineAs'; {onlineAs_N}];
            Test = permute(cpdgen(Uest), [4 1 2 3]);
            imgEst = squeeze(Test(frame, :, :, :));
            whos initAs onlineAs onlineAs_N
        end
        toc;

        testImgErr(t) = frob(imgEst-imgOrg);
        prevImgErr = testImgErr(t);
        writeVideo(outputVideo,uint8(imgEst));
    end
    whos As1 As2 As3 As4
end

close(outputVideo);
fileID = fopen(strcat('OPT/opt', num2str(opt),'.txt'),'w');
testRuntime_Fitness = [testFrame', testRuntime', testFitness', testImgErr', testImgErr1'];
testRuntime_Fitness = testRuntime_Fitness(1:numOfFrames-tao, :);
result = sprintf('%d\t%.4fs\t%.4f%%\t%.f\t%.f\n', testRuntime_Fitness')
fprintf(fileID, '%s', result);
fclose(fileID);
addpath('../packages/tensorlab_2016-03-28');
addpath('../packages/tensor_toolbox-v3.1');
addpath('../packages/onlineCP');
warning('off', 'all');

% OnlineCP w. split in synthetic data
% OnlineCP_split

currentPath = fileparts(mfilename('fullpath'));
options.Display = false; % Show progress on the command line.
options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
options.Refinement = false;

size_tens = [1000 10 20 30];
dims = [10 20 30 1000];
R = 10;
threshold = 5;

startFrame = 0;
endFrame = 1000;
numOfFrames = endFrame - startFrame;
tao = 5;
iterFrame = 5;
T = NaN(dims);
minibatchSize = 1;

for i = startFrame/iterFrame:endFrame/iterFrame-1
    tensorFile = fopen(strcat(currentPath, '/data/data', num2str(i), '.tensor'), 'r');
    tic;
    X = fscanf(tensorFile, "%d %d %d %d %f", [5, inf]);
    for row = X
        T(row(2), row(3), row(4), row(1)) = row(5);
    end
    fclose(tensorFile);
    toc;
end


idx = repmat({':'}, 1, length(dims));
prevDrasticFrame = 1;

for frame = 1:minibatchSize:numOfFrames
    t = frame;
    % fprintf('\n> %dth frame\n', frame+startFrame);
    endTime = min(frame+minibatchSize-1, numOfFrames);
    idx(end) = {frame:endTime};
    x = squeeze(T(idx{:}));

    idx(end) = {prevDrasticFrame:endTime};
    Xt = T(idx{:});
    imgOrg = squeeze(Xt(:, :, :, frame-prevDrasticFrame+1));

    tic;

    if frame - prevDrasticFrame < tao-1
        % disp('continue.');
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        continue;

    elseif frame-prevDrasticFrame+1 == tao
        % disp('OnlineCP init! CP-ALS update!');
        initAs = cpd(Xt, R, options);
    
        [onlinePs, onlineQs] = onlineCP_initial_tenlab(Xt, initAs, R);
        onlineAs = initAs(1:end-1);
        onlineAs_N = initAs{end};
    
        Uest = [onlineAs'; {onlineAs_N}];
        Test = cpdgen(Uest);

        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;

        prevImgNormErr = 0;
        for i = 1:tao
            prevFrame = frame-prevDrasticFrame+1-tao+i;
            imgEst = squeeze(Test(:, :, :, prevFrame));
            imgOrg = squeeze(Xt(:, :, :, prevFrame));
            if i > 1
                testImgNormErr1(t-tao+i) = frob(imgEst-imgOrg);
            end
            testImgNormErr(t-tao+i) = frob(imgEst-imgOrg);
            if prevImgNormErr < testImgNormErr(t-tao+i)
                prevImgNormErr = testImgNormErr(t-tao+i);
            end
        end
        continue;

    else
        % disp('OnlineCP update!');
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
        % [prevImgNormErr, testImgNormErr1(t)]
    
        if prevImgNormErr*threshold < testImgNormErr1(t)
            disp(strcat(num2str(frame), 'Drastic scene detected. CP-ALS update triggered!'));
            prevDrasticFrame = frame;
            testFrame(t) = frame+startFrame;
            testRuntime(t) = toc;
            prevImgNormErr = 0;
        end
    end
    % toc;

    testFrame(t) = frame+startFrame;
    testRuntime(t) = toc;
    testImgNormErr(t) = frob(imgEst-imgOrg);
    if prevImgNormErr < testImgNormErr(t)
        prevImgNormErr = testImgNormErr(t);
    end
end
whos As1 As2 As3 As4

fileID = fopen('result.txt','w');
testRuntime_Fitness = [testFrame', testRuntime', testImgNormErr', testImgNormErr1'];
result = sprintf('%d\t%.4f\t%.f\t%.f\n', testRuntime_Fitness');
fprintf(fileID, '%s', result);
fclose(fileID);

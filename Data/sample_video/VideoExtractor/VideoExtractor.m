
videoName = 'video.mp4';

obj = VideoReader(videoName);
get(obj);
disp(obj.NumFrames);

for frame = 1: obj.NumFrames
    fp;
    if mod(frame, 5) == 1
        tensorName = strcat('video', num2str(floor(frame/5)), '.tensor')
        fp = fopen(tensorName, 'w');
    end
    % filename = strcat('video_frame', num2str(frame), '.jpg');
    img = read(obj, frame);
    imgSize = size(img);
    for y_pixel = 1: imgSize(1)
        for x_pixel = 1: imgSize(2)
            for rgb24 = 1: imgSize(3)
                entry = sprintf('%d\t%d\t%d\t%d\t%d', frame, y_pixel, x_pixel, rgb24, img(y_pixel, x_pixel, rgb24));
               % disp(entry);
                fprintf(fp, "%s\n", entry);
            end
        end
    end
    if mod(frame, 5) == 0
        disp('closed');
        fclose(fp);
    end
    % disp(imgSize(1));
    % imwrite(img, filename);
end



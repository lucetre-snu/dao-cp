
videoName = 'test2M.mp4';
tensorName = 'test2M.tensor';

obj = VideoReader(videoName);
get(obj);
% disp(obj.NumFrames);

fp = fopen(tensorName, 'w');
for frame = 1: obj.NumFrames;
   % filename = strcat('OnlineCP-opt/VideoExtractor/frames/frame', num2str(frame), '.jpg');
    img = read(obj, frame);
    imgSize = size(img);
    for y_pixel = 1: imgSize(1);
        for x_pixel = 1: imgSize(2);
            for rgb24 = 1: imgSize(3);
                entry = sprintf('%d\t%d\t%d\t%d\t%d', frame, y_pixel, x_pixel, rgb24, img(y_pixel, x_pixel, rgb24));
               % disp(entry);
                fprintf(fp, "%s\n", entry);
            end
        end
    end
   % disp(imgSize(1));
   % imwrite(img, filename);
end


fclose(fp);

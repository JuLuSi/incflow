%% Load File 1

filename = '/home/juan/projects/lab/incflow/tests/mit81_points.csv';
delimiter = ' ';
startRow = 2;

formatSpec = '%f%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
time = dataArray{:, 1};
Tp1 = dataArray{:, 2};
up1 = dataArray{:, 3};
vp1 = dataArray{:, 4};
nu0 = dataArray{:, 5};
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

%% Load File 2

filename = '/home/juan/projects/lab/incflow/tests/temperature.point';
delimiter = '\t';

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%*s%s%*s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric strings to numbers.
% Replace non-numeric strings with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end


%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Allocate imported array to column variable names
timeref = cell2mat(raw(:, 1));
uref = cell2mat(raw(:, 2));
Tref = cell2mat(raw(:, 3));
nuref = cell2mat(raw(:, 4));


%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me R;%%

mean_Tp1 = mean(Tp1);
mean_up1 = mean(up1);
mean_vp1 = mean(vp1);
mean_nu0 = mean(nu0);

idx = timeref <= 100.0;

mean_Tref = mean(Tref(idx));
mean_uref = mean(uref(idx));
mean_nuref = mean(nuref(idx));

figure();
hold on;
plot(time, Tp1);
plot(timeref(idx), Tref(idx));

figure();
hold on;
plot(time, up1);
plot(timeref(idx), uref(idx));

figure();
hold on;
plot(time, nu0);
plot(timeref(idx), -nuref(idx));

figure();

subplot(4,1,1);
plot(time, Tp1);
ylabel('Temperature');
xlabel('t');
title('Temperature p1');

subplot(4,1,2);
plot(time, up1);
ylabel('VelX');
xlabel('t');
title('VelX p1');

subplot(4,1,3);
plot(time, vp1);
ylabel('VelY');
xlabel('t');
title('VelY p1');

subplot(4,1,4);
plot(time, -nu0);
ylabel('nu');
xlabel('t');
title('Nusselt x=0');
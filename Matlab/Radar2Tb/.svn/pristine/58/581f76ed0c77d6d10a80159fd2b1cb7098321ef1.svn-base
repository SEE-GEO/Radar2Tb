% FORMAT [ye,mo,da,ho,mi] = get_time(P)
%
% IN    P   Path structure
% OUT   ye  Year
%       mo  Month
%       da  Day
%       ho  Hour
%       mi  Minute

% 2020-12-25 Patrick Eriksson


function [ye,mo,da,ho,mi] = get_time(P)

s = xmlLoad( fullfile( P.wfolder, 'time.xml' ) );

ye = str2num( s(1:4) );
mo = str2num( s(5:6) );
da = str2num( s(7:8) );
ho = str2num( s(9:10) );
mi = str2num( s(10:11) );

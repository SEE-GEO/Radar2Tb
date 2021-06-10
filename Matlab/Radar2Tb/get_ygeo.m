% FORMAT y_geo = get_ygeo(P,C)
%
% OUT   y_geo  As ARTS *y_geo*, but thinned to only have one row per simulation
% IN    P      Path structure
%       C      Calculation settings structure

% 2020-12-18 Patrick Eriksson

function y_geo = get_ygeo(P,C)

if ~C.do_csky
  error( 'y_geo not produced by an all-sky calculation.' );
end

y_geo  = xmlLoad( fullfile(P.wfolder,'y_geo.xml') );
f_grid = xmlLoad( fullfile(P.wfolder,'f_grid.xml') );
stokes = xmlLoad( fullfile(P.wfolder,'stokes_dim.xml') );

y_geo = y_geo(1:length(f_grid)*stokes:end,:);


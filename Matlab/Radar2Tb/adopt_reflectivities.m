% FORMAT adopt_dardar_iwc(P)
% 
% IN    P   Path structure
%
% This functions make use of
%     reflectivities.xml 
% and sets
%     dbze
%     cloudbox_limits

% 2021-01-11 Patrick Eriksson


function adopt_reflectivities(P)


%- Load p_grid as basis for altitude cropping
%
p_grid = xmlLoad( fullfile( P.wfolder, 'p_grid.xml' ) );


% Reflectivities shall be expanded with -99
% They are coming as Tensor4
%
T = xmlLoad( fullfile( P.wfolder, 'reflectivities.xml' ) );
%
% Crop
T = squeeze( T );
T = T(1:length(p_grid),:,:);
%
% Convert to dBZe
fillvalue = -99;  %dBZe
zlim = 10^(fillvalue/10);
T(T<zlim) = zlim;
T = 10*log10(T);
%
% Expand w.r.t. cloudbox
T2 = repmat( fillvalue, size(T)+[0 4] );
T2(:,3:end-2) = T;


% Store
%
xmlStore( fullfile( P.wfolder, 'reflectivities.xml' ), T2, 'Tensor3', 'binary' );

% FORMAT set_z_surface(P)
%
% IN    P   Path structure
%
% This functions make use of
%     z_surface
% and sets
%     z_surface

% 2020-12-19 Patrick Eriksson


function set_z_surface(P)


%- z_surface
%
z = xmlLoad( fullfile( P.wfolder, 'z_surface.xml' ) );
%
z = [z(1);z(1);z;z(end);z(end)];
%
xmlStore( fullfile( P.wfolder, 'z_surface.xml' ), z, 'Matrix', 'binary' );


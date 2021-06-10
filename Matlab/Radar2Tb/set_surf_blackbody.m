% FORMAT set_surf_blackbody(P)
%
% IN    P   Path structure
%
% This functions make use of
%     -
% and sets
%     surface.arts
%     surface_type_mask

% 2020-12-26 Patrick Eriksson


function set_surf_blackbody(P)


%- Copy include file to use
%
copyfile( fullfile( P.arts_files, 'surface_blackbody.arts' ), ...
          fullfile( P.wfolder, 'surface.arts' ) );

% Surface skin temperature is taken from t_field 


%- Surface type mask (land everywhere)
%  (not used by ARTS, but by *get_beam_info*)
%
Gf.name      = 'Surface type';
Gf.gridnames = { 'Latitude', 'Longitude' };
Gf.grids     = { [-90 90], [-180 360] };
Gf.dataname  = 'Data';
Gf.data      = ones(2,2);
%
xmlStore( fullfile( P.wfolder, 'surface_type_mask.xml' ), Gf, 'GriddedField2' );

% FORMAT set_surf_fastem(P)
%
% IN    P   Path structure
%
% This functions make use of
%     skt.xml
%     wind_speed
%     wind_direction
% and sets
%     z_surface
%     surface.arts
%     skin_t_field
%     surface_wind_speed
%     surface_wind_direction
%     surface_type_mask

% 2020-12-18 Patrick Eriksson


function set_surf_fastem(P)

%- Set z_surface to 0 everywhere
%
lat = xmlLoad( fullfile( P.wfolder, 'lat_grid.xml' ) );
%
xmlStore( fullfile( P.wfolder, 'z_surface.xml' ), zeros(size(lat)), ...
          'Matrix', 'binary' );


%- Copy include file to use
%
copyfile( fullfile( P.arts_files, 'surface_fastem.arts' ), ...
          fullfile( P.wfolder, 'surface.arts' ) );


%- Recreate original latitude grid 
%
lats = xmlLoad( fullfile( P.wfolder, 'lat_true.xml' ) );
lats = lats(3:end-2);
%
% Can be in reversed order and we need to resort all data that go into GF2-s.
[lats,ind] = sort( lats );


%- Skin temperature 
%
d    = xmlLoad( fullfile( P.wfolder, 'skt.xml' ) );
d    = d(ind);
%
Gf.name      = 'Surface skin temperature';
Gf.gridnames = { 'Latitude', 'Longitude' };
Gf.grids     = { lats, [-180 360] };
Gf.dataname  = 'Data';
Gf.data      = [d d];
%
xmlStore( fullfile( P.wfolder, 'skin_t_field.xml' ), Gf, ...
          'GriddedField2', 'binary' );


%- Wind speed
%
d = xmlLoad( fullfile( P.wfolder, 'wind_speed.xml' ) );
d = d(ind);
Gf.name  = 'Wind speed';
Gf.data  = [d d];
%
xmlStore( fullfile( P.wfolder, 'surface_wind_speed.xml' ), Gf, ...
          'GriddedField2', 'binary' );


%- Wind direction
%
d = xmlLoad( fullfile( P.wfolder, 'wind_direction.xml' ) );
d = d(ind);
Gf.name  = 'Wind direction';
Gf.data  = [d d];
%
xmlStore( fullfile( P.wfolder, 'surface_wind_direction.xml' ), Gf, ...
          'GriddedField2', 'binary' );


%- Surface type mask (water everywhere)
%  (not used by ARTS, but by *get_beam_info*)
%
Gf.name      = 'Surface type';
Gf.grids     = { [-90 90], [-180 360] };
Gf.data      = zeros(2,2);
%
xmlStore( fullfile( P.wfolder, 'surface_type_mask.xml' ), Gf, ...
          'GriddedField2', 'binary' );

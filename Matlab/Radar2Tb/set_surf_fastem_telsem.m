% FORMAT set_surf_fastem_telsem(P)
%
% IN    P   Path structure
%
% This functions make use of
%     lsm
%     skt.xml
%     wind_speed
%     wind_direction
% and sets
%     surface.arts
%     skin_t_field
%     surface_wind_speed
%     surface_wind_direction
%     land_sea_mask
%     telsem_folder
%     telsem_month

% 2020-12-19 Patrick Eriksson


function set_surf_fastem_telsem(P)


%- Copy include file to use
%
copyfile( fullfile( P.arts_files, 'surface_fastem_telsem.arts' ), ...
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
Gf.name = 'Wind speed';
Gf.data = [d d];
%
xmlStore( fullfile( P.wfolder, 'surface_wind_speed.xml' ), Gf, ...
          'GriddedField2', 'binary' );


%- Wind direction
%
d = xmlLoad( fullfile( P.wfolder, 'wind_direction.xml' ) );
d = d(ind);
Gf.name = 'Wind direction';
Gf.data = [d d];
%
xmlStore( fullfile( P.wfolder, 'surface_wind_direction.xml' ), Gf, ...
          'GriddedField2', 'binary' );


%- Land-sea mask
%
d = xmlLoad( fullfile( P.wfolder, 'lsm.xml' ) );
d = d(ind);
Gf.name = 'Surface type';
Gf.data = [d d];
%
xmlStore( fullfile( P.wfolder, 'surface_type_mask.xml' ), Gf, ...
          'GriddedField2', 'binary' );


%- Telsem
%
[~,month] = get_time( P );
%
xmlStore( fullfile( P.wfolder, 'telsem_folder.xml' ), P.telsem, 'String' );
xmlStore( fullfile( P.wfolder, 'telsem_month.xml' ), month, 'Index' );

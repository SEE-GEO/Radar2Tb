% FORMAT adopt_era5_cksy(P,O)
%
% IN    P   Path structure
%       O   Calculation options structure. See *o_std* for field descriptions.
%
% The ERA5 zip is unpacked.
% This functions adopts/sets:
%     lat_true
%     lon_true
%     lat_grid
%     p_grid
%     t_field
%     z_field
%     vmr_field
%
% These fields of *O* are applied:
%   All pressure levels above O.z_toa are removed
%   All LWC below O.lwc_min are set to zero
%   All LWC at temperatures below O.lwc_tmin are set to zero

% 2020-12-18 Patrick Eriksson


function adopt_era5_csky(P,O)

% ARTS does not allow that particles are found at the lat or lon edge of the
% atmosphere. Nor at the edge of the cloud box. For this reason we must expand
% all fields with a clear-sky part, in both ends. Two new points are needed in
% each end. This means that we need to process most fields, before they can be
% used by ARTS for all-sky simulations.


%- Unpack ERA5 data
%
zip_file = fullfile( P.wfolder, 'era5.zip' );
copyfile( P.era5_zip, zip_file );
system( sprintf( 'unzip -qq -o -d %s %s', P.wfolder, zip_file ) );
delete( zip_file );


%- Latitudes and longitudes
%
% The ERA5 latitudes map here to lat_true in ARTS. For 2D, the ARTS
% lat_grid is an angle along the orbit.
%
lat_true = xmlLoad( fullfile( P.wfolder, 'lat_grid.xml' ) );
lon_true = xmlLoad( fullfile( P.wfolder, 'lon_grid.xml' ) );
%
nlat = length( lat_true );
%
% We assume that the distance is the same between all points
% Take the distance over 10 profiles to get better precision
% We add [dd 30] degrees in each end
% For lat/lon_true, we can just repeat end values
n  = min( 10, nlat );
dd = sphdist( lat_true(1), lon_true(1), lat_true(1+n), lon_true(1+n) ) / n;
%
lat_grid = [-30 -dd dd*(0:nlat-1) dd*nlat+[0 30]]';
lat_true = [lat_true(1);lat_true(1);lat_true;lat_true(end);lat_true(end)];
lon_true = [lon_true(1);lon_true(1);lon_true;lon_true(end);lon_true(end)];
%
xmlStore( fullfile( P.wfolder, 'lat_grid.xml' ), lat_grid, 'Vector', 'binary' );
xmlStore( fullfile( P.wfolder, 'lat_true.xml' ), lat_true, 'Vector', 'binary' );
xmlStore( fullfile( P.wfolder, 'lon_true.xml' ), lon_true, 'Vector', 'binary' );


% z_field
%
% We expand by copying edge values
%
T = xmlLoad( fullfile( P.wfolder, 'z_field.xml' ) );
%
% Find altitudes to keep
zm = mean( T, 2 );
iz = find( zm <= O.z_toa );
%
T = [T(iz,1) T(iz,1) T(iz,:) T(iz,end) T(iz,end)];
xmlStore( fullfile( P.wfolder, 'z_field.xml' ), T, 'Tensor3', 'binary' );


%- p_grid
%
p_grid = xmlLoad( fullfile( P.wfolder, 'p_grid.xml' ) );
xmlStore( fullfile( P.wfolder, 'p_grid.xml' ), p_grid(iz), 'Vector', 'binary' );


%- Other fields
%
T = xmlLoad( fullfile( P.wfolder, 't_field.xml' ) );
T = T(iz,:);
Tt = T;
T = [T(:,1) T(:,1) T T(:,end) T(:,end)];
xmlStore( fullfile( P.wfolder, 't_field.xml' ), T, 'Tensor3', 'binary' );
%
T = xmlLoad( fullfile( P.wfolder, 'vmr_field.xml' ) );
T = T(:,iz,:);
T2 = zeros( size(T) + [0 0 4] );
%
ilwc = 5;
%
for i = 1 : size(T,1)
  This = T(i,:,:);
  % Filter LWC
  if i == ilwc
    This(This<O.lwc_min) = 0;
    This(Tt<O.lwc_tmin) = 0;
  end
  T2(i,:,1) = This(:,1);
  T2(i,:,2) = This(:,1);
  T2(i,:,3:end-2) = This;
  T2(i,:,end-1) = This(:,end);
  T2(i,:,end) = This(:,end);
end
%
xmlStore( fullfile( P.wfolder, 'vmr_field.xml' ), T2, 'Tensor4', 'binary' );
%
% And a check:
T = xmlLoad( fullfile( P.wfolder, 'abs_species.xml' ) );
assert( strcmp( T{1}, 'N2' ) );
assert( strcmp( T{2}, 'O2' ) );
assert( strcmp( T{3}, 'H2O' ) );
assert( strcmp( T{4}, 'O3' ) );
assert( strcmp( T{ilwc}, 'LWC' ) );
%
clear T T2

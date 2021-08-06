% FORMAT set_poslos(P,z,incang,lstep,inc)
% 
% IN    P        Path structure
%       z        Simulation altitude
%       incang   Surface incidence angle
%       lstep    Length between each simulation
%       inc      True if going in order of increasing lat_grid 
%
% Sets
%    sensor_pos
%    sensor_los

% 2020-12-18 Patrick Eriksson


function set_poslos(P,z,incang,lstep,inc)

dlat = lstep / 111e3;
re   = constants('EARTH_RADIUS');
za   = 180 - asind( re*sind(incang) / (re+z) );      

lat_grid = xmlLoad( fullfile( P.wfolder, 'lat_grid.xml' ) );


if inc
   lat = lat_grid(3) : dlat : (lat_grid(end-2)-(2.5*z/111e3));
else
   lat = (lat_grid(3)+(2.5*z/111e3)) : dlat : lat_grid(end-2);
   za = -za;
end

sensor_pos = zeros( length(lat), 2 );
sensor_pos(:,1) = z;
sensor_pos(:,2) = lat;

sensor_los = repmat( za, length(lat), 1 );

xmlStore( fullfile( P.wfolder, 'sensor_pos.xml' ), sensor_pos, 'Matrix', 'binary' );
xmlStore( fullfile( P.wfolder, 'sensor_los.xml' ), sensor_los, 'Matrix', 'binary' );



% FORMAT B = get_beam_info(P,C)
%
% The output B structure has fields:
%    pos    sensor_pos
%    lat    Latitude, at surface
%    lon    Longitude, at surface
%    stype  Surface type
%    z0     Altitude of ground intersection
%    p0     Pressure at ground intersection
%    t0     Air temperature at ground intersection
%    t2m    2m temperatur at ground intersection
%    tskin  Surface skin temperature at ground intersection
%    iwp    Ice water path
%    rwp    Rain water path
%    wvp    Water wapour path
%
% OUT   B      Beam data structure
% IN    P      Path structure
%       C      Calculation settings structure

% 2020-12-19 Patrick Eriksson


function B = get_beam_info(P,C)

if C.do_csky
  error( 'This method requires that an all-sky calculation has been done.' );
end

% Set up output arguments
%
B.pos    = xmlLoad( fullfile(P.wfolder,'sensor_pos.xml') ); 
npos     = size( B.pos, 1 );
B.lat    = zeros( npos, 1 );
B.lon    = zeros( npos, 1 );
B.stype  = zeros( npos, 1 );
B.z0     = zeros( npos, 1 );
B.p0     = zeros( npos, 1 );
B.t0     = zeros( npos, 1 );
B.t2m    = zeros( npos, 1 );
B.tskin  = zeros( npos, 1 );
B.iwp    = zeros( npos, 1 );
B.rwp    = zeros( npos, 1 );
B.wvp    = zeros( npos, 1 );


% Check that water vapour is at expected position
%
ih2o = 3;    % 1-based
%
abs_species = xmlLoad( fullfile(P.wfolder,'abs_species.xml') ); 
%
if ~strcmp( abs_species{ih2o}, 'H2O' )
  error( 'Water vapour expected to be abs_species nr %d', ih2o );
end
%
h2o_string = sprintf( 'VMR species %d', ih2o-1 );
avog_const = constants( 'AVOGADRO' );


% Process ATM-1D data
%
any_iwp = false;
%
for i = 1 : npos
  A = xmlLoad( fullfile(P.wfolder,sprintf('atm1d.%04d001000.xml',i)) );
  B.lat(i) = A.grids{3};
  B.lon(i) = A.grids{4};
  B.z0(i)  = A.data(1,1);
  B.t0(i)  = A.data(2,1);
  B.p0(i)  = A.grids{2}(1);
  %
  for f = 1 : length(A.grids{1})
    % IWP and RWP
    if strncmp( A.grids{1}{f}, 'Mass category', 13 )
      imass = 1 + str2num( A.grids{1}{f}(14:end) );
      if imass == 1
        B.rwp(i) = trapz( A.data(1,:), A.data(f,:) );
      elseif imass == 2
        B.iwp(i) = trapz( A.data(1,:), A.data(f,:) );
        any_iwp = true;
      else
        error( 'Mass category %.0f found. Not handled.', imass );
      end
    end
    % WVP
    if strcmp( A.grids{1}{f}, h2o_string )
      nd = vmr2nd( A.data(f,:), A.grids{2}', A.data(2,:) );
      B.wvp(i) = (0.018015/avog_const) * trapz( A.data(1,:), nd );
    end
  end
end
%
if ~any_iwp
  B.iwp = B.rwp;
  B.rwp = zeros( npos, 1 );
end


% Get surface types
%
T = xmlLoad( fullfile(P.wfolder,'surface_type_mask.xml') );
%
B.stype = interp1( T.grids{1}, T.data(:,1), B.lat, 'nearest' );


% Get surface skin temperature
%
T = xmlLoad( fullfile(P.wfolder,'skin_t_field.xml') );
%
B.tskin = interp1( T.grids{1}, T.data(:,1), B.lat );


% Get 2m temperature
%
lat = xmlLoad( fullfile(P.wfolder,'lat_true.xml') );
t2m = xmlLoad( fullfile(P.wfolder,'t2m.xml') );
%
B.t2m = interp1( lat(3:end-2), t2m, B.lat );






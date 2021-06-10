% RUN1GMI   GMI simulations over one orbit part
%
% The fields of *P* are:
%   P.outfolder   Folder where to save results
%   P.era5_zip    Full path to input zip file
%   P.wfolder     Work folder. If empty, a temporary folder is created.
%   P.arts        Full path to ARTS executable.
%   P.arts_files  Full path to input files coming with Radar2Tb
%   P.std_habits  Full path to folder with standard habits files.
%   P.telsem      Full path to folder with TELSEM and TESSEM input files.
%
% FORMAT run1gmi(P,O)
%
% IN   P   See above
%      O   Calculation options structure. See *o_std* for field descriptions.

% 2021-01-11   Patrick Eriksson

function run1gmi(P,O)


%- Create name of output file
%
[~,casename] = fileparts( P.era5_zip );
outfile = fullfile( P.outfolder, [casename,'.mat'] );


%- Check input
%
if exist( outfile, 'file' )
  fprintf( 'Output file already exists (%s).\n', casename );
  return
end
if ~exist( P.era5_zip, 'file' )
  fprintf( 'Did not find\n%s\n', P.era5_zip );
  return
end
%
fprintf( 'Doing %s ...', casename );


%- Workfolder
%
if isempty( P.wfolder )
  P.wfolder = create_tmpfolder;
  cu = onCleanup( @()delete_tmpfolder( P.wfolder ) );
end


%- Hard-coded settings
%
C.arts_time  = false;


%- Common stuff
%
adopt_era5_csky( P, O );
copyfile( fullfile(P.arts_files,O.rainpsd), ...
          fullfile(P.wfolder,'psd_rwc.arts') );
copyfile( fullfile(P.arts_files,O.icepsd), ...
          fullfile(P.wfolder,'psd_iwc.arts') );
set_z_surface( P );
set_surf_fastem_telsem_snow( P );


%- Onion peeling part
%
adopt_reflectivities( P );
%
C = set_cloudsat( P, C );
set_habit( P, C, O.icehabit, 'LiquidSphere', O.icesize, O.pratio_csat );
%
run_onion( P, O );


%- Init GMI
%
C.pol_mode   = 'V';  
incang = set_gmi( P, C );
set_poslos( P, O.z_toa+1e3, incang, O.lsampling, true );
set_rt4( P, 12 );


%- Make a clear-sky run to get incidence angle and surface_rmatrix
%
if O.do_emissivities
  C.do_csky = true;
  run_arts( P, C );
  %
  y_geo  = xmlLoad( fullfile(P.wfolder,'y_geo.xml') );
  f_grid = xmlLoad( fullfile(P.wfolder,'f_grid.xml') );
  stokes = xmlLoad( fullfile(P.wfolder,'stokes_dim.xml') );
  S.incang = y_geo(1:length(f_grid)*stokes:end,4);
  %
  n = length(S.incang);
  S.e166h = zeros( n, 1 );
  S.e166v = zeros( n, 1 );
  S.e183h = zeros( n, 1 );
  S.e183v = zeros( n, 1 );
  %
  for i = 1 : n
    R = xmlLoad( fullfile(P.wfolder,sprintf('R.%04d001000.xml',i)) );
    % Remove angle dimension
    R = getdims( R, [2:4] );
    % Interpolate in frequency
    R = interp1( f_grid, R, [166e9 183e9], 'linear', 'extrap' );
    assert( abs(R(1,1,2)-R(1,2,1)) < 0.001 );
    assert( abs(R(2,1,2)-R(2,2,1)) < 0.001 );
    S.e166h(i) = 1 - ( R(1,1,1) - R(1,1,2));
    S.e166v(i) = 1 - ( R(1,1,1) + R(1,1,2));
    S.e183h(i) = 1 - ( R(2,1,1) - R(2,1,2));
    S.e183v(i) = 1 - ( R(2,1,1) + R(2,1,2));
  end
else
  S = NaN;
end


%- GMI, V channels
%
% set_gmi called above
C.do_csky = false;
set_habit( P, C, O.icehabit, 'LiquidSphere', O.icesize, O.pratio_gmi );
run_arts( P, C );
TbV = get_tb_gmi( P, C );


%- GMI, H channels 
%
C.pol_mode   = 'H';
set_gmi( P, C );
set_habit( P, C, O.icehabit, 'LiquidSphere', O.icesize, O.pratio_gmi );
run_arts( P, C );
TbH = get_tb_gmi( P, C );


%- Combine V and H
%
Tb      = TbV;
Tb(:,2) = TbH(:,2);


%- Get information for each pencil beam
%
B = get_beam_info( P, C);


%- Save to outfile
%
save( outfile, 'Tb', 'O', 'B', 'S' );
%
fprintf( '\rDone: %s      \n', casename );

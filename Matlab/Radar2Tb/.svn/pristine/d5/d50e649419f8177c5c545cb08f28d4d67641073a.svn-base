% Basic demonstration of the function set

P.era5_zip   = '/home/patrick/Tmp/2009_182_02_A.zip';
P.wfolder    = '/home/patrick/WORKAREA';
P.arts       = '/home/patrick/ARTS/arts/build/src/arts';
P.arts_files = '/home/patrick/SVN/opengem/patrick/Projects/Radar2Tb/ArtsFiles';
P.std_habits = '/home/patrick/Data/StandardHabits';
P.telsem     = '/home/patrick/Try/IWP/TessemTelsem';

F.z_toa      = 25e3;    % Cut atmosphere around this altitude
F.lwc_min    = 1e-7;    % LWC below this value is set to 0
F.lwc_tmin   = 273-30;  % LWC at temperatures below this value is set to 0
F.iwc_min    = 1e-8;    % IWC below this value is set to 0
                        %
F.phase_tlim = 273;     % No RWC/IWC below/above this temperature

C.do_csky    = true;
C.pol_mode   = 'I';
C.arts_time  = true;

adopt_era5_csky( P, F );
set_surf_fastem_telsem( P );
incang = set_gmi( P, C );
set_poslos( P, F.z_toa, incang, 50e3, true );

if C.do_csky
  run_arts( P, C );
else
  adopt_dardar_iwc( P, F );
  copyfile( fullfile(P.arts_files,'psd_no_rwc.arts'), ...
          fullfile(P.wfolder,'psd_rwc.arts') );
  if 0
    set_habit( P, C, 'LargePlateAggregate', [], 'dmax' );
    copyfile( fullfile(P.arts_files,'psd_f07t.arts'), ...
              fullfile(P.wfolder,'psd_iwc.arts') );
  else
    set_habit( P, C, 'LargePlateAggregate', [], 'dveq' );
    copyfile( fullfile(P.arts_files,'psd_dardar1mom.arts'), ...
              fullfile(P.wfolder,'psd_iwc.arts') );
  end
  set_rt4( P, 12 );
  run_arts( P, C );
end

%- Plot
%
Tb = get_tb_gmi( P, C );
%
if C.do_csky
  y_geo = get_ygeo( P, C );
  lat_values = y_geo(:,2);
  lat_true = xmlLoad( fullfile( P.wfolder, 'lat_true.xml' ) );
  lat_grid = xmlLoad( fullfile( P.wfolder, 'lat_grid.xml' ) );
  lat = interp1( lat_grid, lat_true, lat_values );
else
  [iwp,rwp,lat] = get_iwp( P, C );  
end
%
figure(1)
plot( lat, Tb' );
xlabel( 'Latitude of ground intersection' );
ylabel( 'Tb [K]' );
legend( '166V', '166H', '183+-7', '183+-3' );
%
if ~C.do_csky
  figure(2)
  plot( lat, iwp );
  xlabel( 'Latitude of ground intersection' );
  ylabel( 'IWP [kg/m2]' );
end

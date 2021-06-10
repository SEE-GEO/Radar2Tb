% FORMAT run_onion(P,O)
%
% IN   P   Path structure
%      O   Calculation options structure. See *o_std* for field descriptions.
%
% These fields of *O* are provided to ARTS:
%     O.phase_tlim
%     O.onion_dBZe_noise  
%     O.onion_fill_clutter
%     O.onion_h_clutter   
%     O.onion_hyd_max     
%     O.onion_hyd_scaling 
%     O.onion_wc_clip     
%     O.onion_wc_max      

% 2020-12-19 Patrick Eriksson

function run_onion(P,O)


%- Set control file
%
cfile = fullfile( P.wfolder, 'cfile.arts' );
%
copyfile( fullfile( P.arts_files, 'onion.arts' ), cfile );


%- Set h_clutter vector
%
lat   = xmlLoad( fullfile( P.wfolder, 'lat_true.xml' ) );
STM   = xmlLoad( fullfile( P.wfolder, 'surface_type_mask.xml' ) );
stype = interp1( STM.grids{1}, STM.data(:,1), lat, 'nearest' );
%
h_clutter = repmat( O.h_clutter_ocean, length(lat), 1 );
h_clutter(stype > 0) = O.h_clutter_land;
h_clutter = boxcarfilter( 1:length(h_clutter), h_clutter, 3 );
%
xmlStore( fullfile( P.wfolder, 'h_clutter.xml' ), h_clutter, ...
          'Matrix', 'binary' );


%- Special saves
%
xmlStore( fullfile( P.wfolder, 'phase_tlim.xml' ), O.phase_tlim, 'Numeric' );
%
xmlStore( fullfile( P.wfolder, 'onion_dBZe_noise.xml' ), O.onion_dBZe_noise, ...
          'Numeric' );
xmlStore( fullfile( P.wfolder, 'onion_fill_clutter.xml' ), O.onion_fill_clutter, ...
          'Index' );
xmlStore( fullfile( P.wfolder, 'onion_hyd_max.xml' ), O.onion_hyd_max, ...
          'Numeric' );
xmlStore( fullfile( P.wfolder, 'onion_hyd_scaling.xml' ), O.onion_hyd_scaling, ...
          'Numeric' );
xmlStore( fullfile( P.wfolder, 'onion_wc_clip.xml' ), O.onion_wc_clip, ...
          'Numeric' );
xmlStore( fullfile( P.wfolder, 'onion_wc_max.xml' ), O.onion_wc_max, ...
          'Numeric' );


%- Run ARTS
%
[s,r] = system( sprintf('%s -r000 -o %s %s', P.arts, P.wfolder, cfile ) );
%
if s
  disp( r );
  error( 'Error while running ARTS. See above.' );
end

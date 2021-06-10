% FORMAT incang = set_gmi(P,C) 
%
% OUT   incang Incident angle
% IN    P      Path structure
%       C      Calculation settings structure
%
% Sets
%    f_grid
%    stokes_dim
%    absorption.arts

% 2020-12-18 Patrick Eriksson

function incang = set_gmi(P,C)


%- Incident angle
%
incang = 52.8;


%- f_grid + stokes
%
f_all = 1e9 * [ 166.5, 183.31+[-7 -3 3 7] ]; 

if any( strcmp( C.pol_mode, {'I','V'} ) )
  xmlStore( fullfile( P.wfolder, 'f_grid.xml' ), f_all, 'Vector' );
elseif strcmp( C.pol_mode, 'H' )
  xmlStore( fullfile( P.wfolder, 'f_grid.xml' ), f_all(1), 'Vector' );
else
  error( 'Unknown choice for C.pol_mode (%s)', C.pol_mode );
end


%- stokes
%
xmlStore( fullfile( P.wfolder, 'stokes_dim.xml' ), 2, 'Index' );


%- iy_unit
%
xmlStore( fullfile( P.wfolder, 'iy_unit.xml' ), 'PlanckBT', 'String' );


%- Species and absorption files
%
copyfile( fullfile( P.arts_files, 'continua_rttov.arts' ), ...
          fullfile( P.wfolder, 'continua.arts' ) );
copyfile( fullfile( P.arts_files, 'abs_lines_h2o_rttov_below340ghz.xml' ), ...
          fullfile( P.wfolder, 'abs_lines_h2o_rttov_below340ghz.xml' ) );
copyfile( fullfile(P.arts_files,'absorption_gmi.arts'), ...
          fullfile(P.wfolder,'absorption.arts') );

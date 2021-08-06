% FORMAT Tb = get_tbs_icecube(P)
%
% OUT   Tb     Tb values, one column per position.
% IN    P      Path structure

% 2020-12-26 Patrick Eriksson

function Tb = get_tb_icecube(P)

if 1~= xmlLoad( fullfile(P.wfolder,'stokes_dim.xml') );
  error( 'This function requires that stokes_dim = 1.' );
end

y = xmlLoad( fullfile(P.wfolder,'y.xml') );
f = xmlLoad( fullfile(P.wfolder,'f_grid.xml') );

nf = length(f);
n  = length(y) / nf;

if nf > 1
  Tb = mean( reshape( y, nf, n ) ); 
else
  Tb = y';
end
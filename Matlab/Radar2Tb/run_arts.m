% FORMAT run_arts(P,C)
% 
% IN    P        Path structure
%       C        Calculation settings structure

% 2020-12-19 Patrick Eriksson

function run_arts(P,C)


% Set control file
%
cfile = fullfile( P.wfolder, 'cfile.arts' );
%
if C.do_csky
  copyfile( fullfile( P.arts_files, 'clearsky.arts' ), cfile );
else
  copyfile( fullfile( P.arts_files, 'allsky.arts' ), cfile );
end    


%- Run ARTS
%
if C.arts_time 
  try
    t0 = toc;
  catch
    tic;
    t0 = toc;    
  end
end
%
[s,r] = system( sprintf('%s -r000 -o %s %s', P.arts, P.wfolder, cfile ) );
%
%
if s
  disp( r );
  error( 'Error while running ARTS. See above.' );
end
%
if C.arts_time
  t = toc;
  fprintf( 'ARTS part took %.1f s\n', t-t0 );
end

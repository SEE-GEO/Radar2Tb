% FORMAT set_habit(P,C,iwc_habit,rwc_habit,sort_unit,polratio,thinfac)
%
% IN    P          Path structure
%       C          Calculation settings structure
%       iwc_habit  Name of habit to use for ice
%       rwc_habit  Name of habit to use for liquid. Can be empty.
% OPT   sort_unit  Sort in size according to use unit. Only applied to ice.
%                  Default is 'dveq'. Can alse be 'dmax'
%       polratio   Polarisation ratio, to approx. particle orientation.
%                  Default is 1.
%       iwcthin    Thin data IWC habit, in size, according to this factor.
%                  Default is 1 (no thinning).
%       rwcthin    Thin data RWC habit, in size, according to this factor.
%                  Default is 1 (no thinning).
%
% Sets
%    scat_data_raw.xml
%    scat_meta

% 2020-12-18 Patrick Eriksson


function set_habit(P,C,iwc_habit,rwc_habit,sort_unit,polratio,iwcthin,rwcthin)
%
if nargin < 5 | isempty(sort_unit), sort_unit = 'dveq'; end   
if nargin < 6 | isempty(sort_unit), polratio = 1; end   
if nargin < 7 | isempty(iwcthin), iwcthin = 1; end
if nargin < 8 | isempty(rwcthin), rwcthin = 1; end


% RWC
%
if ~isempty(rwc_habit)
  load( fullfile( P.std_habits, rwc_habit ) );
  %
  o = 0;
  for i = unique( [1:rwcthin:length(S) length(S)] )
    o = o + 1;
    Sa{1}{o} = S(i);
    Ma{1}{o} = M(i);
  end
  iiwc = 2;
else
  iiwc = 1;
end


% IWC
%
load( fullfile( P.std_habits, iwc_habit ) );
%
[S,M] = sort_size( S, M, sort_unit );
%
if polratio ~= 1
  alpha = (polratio-1) / (polratio+1);
  if strcmp( C.pol_mode, 'I' )
    error('Polarisation ratio scaling only allowed for C.pol_mode V and H.');
  elseif strcmp( C.pol_mode, 'V' )
    fac = 1 - alpha;
  elseif strcmp( C.pol_mode, 'H' )
    fac = 1 + alpha;
  else
    error( 'Unknown choice for C.pol_mode (%s)', C.pol_mode );
  end    
  for i = 1: length(S)
    S(i).abs_vec_data = fac * S(i).abs_vec_data;
    S(i).ext_mat_data = fac * S(i).ext_mat_data;
    S(i).pha_mat_data = fac * S(i).pha_mat_data;
  end
end
%
o = 0;
for i = unique( [1:iwcthin:length(S) length(S)] )
  o = o + 1;
  Sa{iiwc}{o} = S(i);
  Ma{iiwc}{o} = M(i);
end




%- Save
%
xmlStore( fullfile( P.wfolder, 'scat_data_raw.xml' ), Sa, ...
          'ArrayOfArrayOfSingleScatteringData', 'binary' );
xmlStore( fullfile( P.wfolder, 'scat_meta.xml' ), Ma, ...
          'ArrayOfArrayOfScatteringMetaData', 'binary' );
return




function [S,M] = sort_size( S, M, sort_unit )

  if strcmp( sort_unit, 'dveq' )
    d = [M.diameter_volume_equ];      
  elseif strcmp( sort_unit, 'dmax' )
    d = [M.diameter_max];
  else
    error( 'Unknown selection for *sort_unit* (%s).', sort_unit ); 
  end
  
  [d,ind] = sort(d);
  
  S = S(ind);
  M = M(ind);
return

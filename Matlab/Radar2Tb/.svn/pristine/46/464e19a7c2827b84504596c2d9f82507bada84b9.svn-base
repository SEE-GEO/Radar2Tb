% FORMAT Tb = get_tb_gmi(P,C)
%
% OUT   Tb     Tb values, one column per position. Rows match (in order):
%                 166V, 166H, 183.31+-7V, 183.31+-3V
% IN    P      Path structure
%       C      Calculation settings structure

% 2020-12-18 Patrick Eriksson

function Tb = get_tb_gmi(P,C)


y = xmlLoad( fullfile(P.wfolder,'y.xml') );


if any( strcmp( C.pol_mode, {'I','V'} ) )
  n = 10;
  Tb = repmat( NaN, length(y)/n, 4 );
  Tb(:,1) = y(1:n:end) + y(2:n:end); 
  Tb(:,3) = ( y(3:n:end) + y(4:n:end) + y(9:n:end) + y(10:n:end) ) / 2;
  Tb(:,4) = ( y(5:n:end) + y(6:n:end) + y(7:n:end) + y(8:n:end) ) / 2; 
  if strcmp( C.pol_mode, 'I' ) 
    Tb(:,2) = y(1:n:end) - y(2:n:end); 
  end
  
elseif strcmp( C.pol_mode, 'H' ) 
  n = 2;
  Tb = repmat( NaN, length(y)/n, 4 );
  Tb(:,2) = y(1:n:end) - y(2:n:end); 

else
  error( 'Unknown choice for C.pol_mode (%s)', C.pol_mode );
end

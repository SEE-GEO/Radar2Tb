% FORMAT set_rt4(P,nstreams,naa,quad_type,pfct_method)
%
% IN    P             Path structure
% OPT   nstreams      As the GIN of RT4Calc with same name
%                     Default is 16.
%       naa           As the pfct_aa_grid_size GIN of RT4Calc
%                     Default is 37.
%       quad_type     As the GIN of RT4Calc with same name
%                     Default is 'L'.
%       pfct_method   As the GIN of RT4Calc with same name
%                     Default is 'median'.

% 2020-12-20 Patrick Eriksson


function set_rt4(P,nstreams,naa,quad_type,pfct_method)
%
if nargin < 2 | isempty(nstreams)
    nstreams = 16;
end
if nargin < 3 | isempty(naa)
    naa = 37;
end
if nargin < 4 | isempty(quad_type)
    quad_type = 'L';
end
if nargin < 5 | isempty(pfct_method)
    pfct_method = 'median';
end

if nstreams < 10 | isodd(nstreams)
    error( 'The number of streams must be av even number >= 10.' );
end


xmlStore( fullfile( P.wfolder, 'rt4_nstreams.xml' ), nstreams, 'Index' );
xmlStore( fullfile( P.wfolder, 'rt4_naa.xml' ), naa, 'Index' );
xmlStore( fullfile( P.wfolder, 'rt4_quad.xml' ), quad_type, 'String' );
xmlStore( fullfile( P.wfolder, 'rt4_pfct.xml' ), pfct_method, 'String' );


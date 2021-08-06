% FORMAT adopt_dardar_iwc(P,F[,use_n0star])
% 
% IN    P            Path structure
%       F            Filter settings structure
% OPT   use_n0star   Flag to include N0star in particle_bulkprop_field.
%                    Default is false.
%
% This functions make use of
%     iwc.xml 
% and sets
%     particle_bulkprop_names
%     particle_bulkprop_field
%     cloudbox_limits
%
% All IWC below F.iwc_min are set to zero
% Note that F.phase_tlim is applied by t_max GIN of PSDs

% 2020-12-18 Patrick Eriksson


function adopt_dardar_iwc(P,F,use_n0star)
%
if nargin < 3 | isempty(use_n0star), use_n0star = false; end

%- Load p_grid as basis for altitude cropping
%
p_grid = xmlLoad( fullfile( P.wfolder, 'p_grid.xml' ) );


% particle_bulkprop_field shall be expanded with zeros
%
Ti = xmlLoad( fullfile( P.wfolder, 'iwc.xml' ) );
%
Ti = Ti(1,1:length(p_grid),:,:);


% Filter IWC
%
Ti(Ti<F.iwc_min) = 0;
%
xmlStore( fullfile( P.wfolder, 'phase_tlim.xml' ), F.phase_tlim, 'Numeric' );


% Add N0star?
%
if use_n0star
  Tn = xmlLoad( fullfile( P.wfolder, 'N0star.xml' ) );
  %
  Tn = Tn(1,1:length(p_grid),:,:);
  %
  T2 = zeros( size(Ti) + [1 0 4] );
  T2(1,:,3:end-2) = Ti;    
  T2(2,:,3:end-2) = Tn;    
else
  T2 = zeros( size(Ti) + [0 0 4] );
  T2(1,:,3:end-2) = Ti;  
end


% Store
%
xmlStore( fullfile( P.wfolder, 'particle_bulkprop_field.xml' ), T2, ...
          'Tensor4', 'binary' );
%
if use_n0star
  xmlStore( fullfile( P.wfolder, 'particle_bulkprop_names.xml' ), ...
            {'IWC','N0star'}, 'ArrayOfString');
else
  xmlStore( fullfile( P.wfolder, 'particle_bulkprop_names.xml' ), ...
            {'IWC'}, 'ArrayOfString');
end
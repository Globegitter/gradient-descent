clc
%% parameter in the image
IWIDTH = 3;
IHEIGHT = 3;
%% parameter in the dictionary
% m/z charge type dim length
sLen = 2000;
% molecular species dim length
mLen = 50;
%% Generate w
spsVal = 1;
gW = abs( randn( mLen, IHEIGHT, IWIDTH) );
gW = gW .* ( gW >= spsVal );
gW0 = zeros( IHEIGHT, IWIDTH );    
%% Generate Dictonary (random normal dist. generation)
spsVal2 = 5;
gD = abs( 5 * randn( sLen, mLen ) );
gD = gD .* ( gD >= spsVal2 );
%% Generate Y   
gY = computePreY( gD, gW, gW0 );

%% fitting without nois
fprintf('Learned paramaters from noise-free synthetic data\n');
tic
[ fitD, fitW ] = DictionaryLearning( Y ,1, 50 );
toc

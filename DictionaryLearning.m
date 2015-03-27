function [ fitD, fitW ] = DictionaryLearning( gY ,lambda, molNum )
% check input sizes
assert( size( gY, 1) > 1 && size( gY, 2 ) > 1 && size( gY, 3 ) > 1 );
% initialize sizes
sLen = size( gY, 1 );
iHei = size( gY, 2 );
iWid = size( gY, 3 );
mLen = 50; %% add manually now!!
lVal = 0; %% add manually now!!

%fit variables initialize
fitD = abs( randn( sLen, mLen ) );
fitW = abs( randn( mLen, iHei, iWid ) );
fitW0 = zeros( iHei, iWid );

%iterated optimization
TOL = 1e-8;
ITLIMIT = 100;
preTarget = -inf;
curTarget = computeObjf( gY, fitD, fitW, fitW0, lVal );
itNum = 1;


while ( abs( curTarget - preTarget ) > TOL ) && ( ITLIMIT <= 100 )
    %%update weights
    prefitW = fitW;
    prefitW0 = fitW0;
    for i = 1:iHei
        for j = 1:iWid
%                 [B FitInfo]= lassoglm( fitD, gY(:,i,j), 'normal', 'Lambda', lVal );
%                 fitW(:,i,j) = B(:,1);
%                 fitW0(i,j) = FitInfo.Intercept;
%                 fprintf( '%d %d\n', i, j );
                [beta0,beta] = coordAscentENet( gY(:,i,j), fitD, lVal, 0, {fitW0(i,j) fitW(:,i,j)} );
                fitW(:,i,j) = beta;
                fitW0(i,j) = beta0;
                objMat1(i,j,itNum) = computeObjf( gY(:,i,j), fitD, fitW(:,i,j), fitW0(i,j), lVal );
        end
    end
    preTarget = curTarget;
    curTarget = computeObjf( gY, fitD, fitW, fitW0, lVal );
    predictY = computePreY( fitD, fitW, fitW0 ); %%compute for dictionary update
    fprintf( 'after lasso, curTarget: %f, preTarget: %f\n', curTarget, preTarget );
    assert( curTarget - preTarget >= 0 );
    
    %%update Dictionary
    prefitD = fitD;
    for k = 1:sLen
        for r = 1:mLen
            a = fitW(r,:,:) .* ( gY(k,:,:) - predictY(k,:,:) + fitD(k,r) * fitW(r,:,:) );
            b = fitW(r,:,:).^2;
            a = squeeze( a );
            b = squeeze( b );
            a = sum( sum( a ) );
            b = sum( sum( b ) );
            if b == 0
                %update predictY first
                predictY(k,:,:) = predictY(k,:,:) - fitD( k, r ) * fitW( r, :, : );
                fitD(k,r) = 0;
            else
                %update predictY first
                predictY(k,:,:) = predictY(k,:,:) - fitD( k, r ) * fitW( r, :, : );
                fitD(k,r) = a / b;
                predictY(k,:,:) = predictY(k,:,:) + fitD( k, r ) * fitW( r, :, : );
            end
        end
    end
    for i = 1:iHei
        for j = 1:iWid
            objMat2(i,j,itNum) = computeObjf( gY(:,i,j), fitD, fitW(:,i,j), fitW0(i,j), lVal );
        end
    end
    preTarget = curTarget;
    curTarget = computeObjf( gY, fitD, fitW, fitW0, lVal );
    itNum = itNum + 1;
    fprintf( 'it num: %d curTarget: %f, preTarget: %f\n', itNum, curTarget, preTarget );
    assert( curTarget - preTarget >= 0 );
end


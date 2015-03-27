function [ObjVal] = computeObjf( gY, fitD, fitW, fitW0, lambdaVal )
%computeObj Compute the objective function in the paper
% lambdaVal should be either 1 value or an array with height*width
    preY = computePreY( fitD, fitW, fitW0  );
    rY = gY - preY;
    penalty = lambdaVal .* reshape( sum( abs( fitW ), 1 ), size( fitW, 2 ), size( fitW, 3 ) );
    ObjVal = -0.5 * sum( ( rY(:) .^2 ) ) - sum( sum( penalty ) );

end


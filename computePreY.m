function preY = computePreY( fitD, fitW, fitW0 )
%computePreY This function computes predicted value of spectrum Y (
%dim=s*w*h ). And fitD has dim s*m, fitW has dim=m*i*j, fitW0 has dim=i*j;
a = size( fitW );
if length( a ) == 3 
% sLen, Height, Width
    preY = zeros( size( fitD, 1 ), a( 2 ), a( 3) );
    for i = 1:a(2)
        for j = 1:a(3)
            preY(:,i,j) = fitD * fitW(:,i,j) + fitW0( i, j );
        end
    end
%in this case, only one data point, so the w*h=1, and Y has dim=s*1,and 
%fitD hasdim s*m fitW has dim=m*1, and fitW0 has dim=1; 
elseif length( a ) == 2  
    
    preY = zeros( size( fitD, 1 ), size( fitW, 2 ) );
    preY = fitD * fitW + fitW0;
end

end

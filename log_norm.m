function z = log_norm(x,skip)
%Compute 1 + log(x)
% column skip will be skiped
% Z = log_norm(X) computes 1 + log(x) for x > 1.

%  Compute the sigmoid of each value of z (z can be a matrix,
%  vector or scalar).

row = size(x,1);
column = size(x,2);
z = zeros(size(x));

for i = 1:row
	for j = 1:column
		if j == skip
			if x(i,j) >= 0.05
				z(i,j) = 1+ log10(x(i,j)*20);
			else
				z(i,j) = x(i,j);
			end
		elseif x(i,j) > 1
			z(i,j) = 1 + log10(x(i,j));
		elseif x(i,j) == 1
			z(i,j) = 1;
		end
	end
end

end
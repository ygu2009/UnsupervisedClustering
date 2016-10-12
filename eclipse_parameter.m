function eclipse_parameter(mu,covar)

[V D] = eig(covar);
    
[D order] = sort(diag(D), 'descend');
D = diag(D);
V = V(:, order);

t = 0:pi/100:2*pi;
e = [cos(t) ; sin(t)];        %# unit circle
VV = V*sqrt(D); %%*[1 0; 0 1];   %# scale eigenvectors
e = bsxfun(@plus, VV*e, [0 0]'); %#' project circle back to orig space


%# plot cov and major/minor axes
hold on
plot(e(1,:)+repmat(mu(1), 1, length(e(1,:))), e(2,:)+repmat(mu(2), 1, length(e(1,:))), 'Color','r','linewidth',2);
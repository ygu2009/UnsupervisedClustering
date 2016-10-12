function MAP_EM_alg(data, cc)

[N,D]=size(data);

% stardarize dataset by making each column be zero mean, standard derivation 1.
y = data - repmat(mean(data), [N 1]);
y = y ./ repmat(std(data), [N 1]);

figure,
set(gcf,'units','points','position',[200,200,1000,300])

subplot(131)
gscatter(y(:,1), y(:,2), cc)
title('Ground Truth')
box on
drawnow;


% initial by using kmeans
K=10; % input your guess

[IDX,C] = kmeans(y,K);

for i=1:K
    mu(i,:)=mean(y(find(IDX(:)==i),:));
    sigma(:,:,i)=(cov(y(find(IDX(:)==i),:)));
	PI(i)=length(y(find(IDX(:)==i),:));
end
PI=PI/sum(PI);

init_mu=mu;
init_sigma=sigma;
init_PI=PI;

% keep the kmeans initials figure
%initials plot
subplot(132)
plot(y(:,1),y(:,2),'b*')
% plot the ellipse for initial cluster
for i=1:K
    hold on
    plot(mu(i,1),mu(i,2),'g*')
    eclipse_parameter(mu(i,:),sigma(:,:,i));
    text(mu(i,1), mu(i,2), num2str(i),'BackgroundColor', [.8 .8 .8]);
end
title('KMeans initials')
drawnow;


% initial likelihood
% Q0=0;
for i=1:N
    for j=1:K
        %probabilty of Gaussian
        p_y(i,j)=(1/((2*pi)*sqrt(det(sigma(:,:,j)))))*exp(-0.5*(y(i,:)-mu(j,:))*inv(sigma(:,:,j))*(y(i,:)-mu(j,:))');
    end
    
end

Q0=sum(log(p_y*PI'));

diff_Q=1;
iter=0;

% MAP EM initial priors
alpha=0.001*ones(1,K);
beta_0=3*ones(1,K);
m0=zeros(K,2);
S0=3*repmat(eye(D),[1 1 K]);
v0=4;
% S0=(K^(1/D))*repmat(inv(diag(var(y))), [1,1,K]); %~~(11.48) Kevin Murphy
% v0=D+2;

% %Regular EM initials
% alpha=1*ones(1,K);
% beta_0=0*ones(1,K);
% m0=0*ones(K,2);
% S0=0*repmat(eye(D),[1 1 K]);
% v0=-D-2;

%% MAP EM steps

while iter<500 & diff_Q>1e-3
    
    iter=iter+1;
    
    % E step
    r=(p_y.*repmat(PI,N,1))./repmat(p_y*PI',1,K);  %unchanged E step (page. 356)
    
    % M step
   
    % estimate mu
    for j=1:K
        temp=[0];
        for i=1:N
            temp=temp+r(i,j)*y(i,:);
        end
        y_bar(j,:)=temp/sum(r(:,j)); %(11.45) from Kevin Murphy
        mu_hat(j,:)=(sum(r(:,j))*y_bar(j,:)+beta_0(j)*m0(j))/(sum(r(:,j))+beta_0(j)); %(11.43) from Kevin Murphy
    end
    
    
    % estimate sigma
    
    for j=1:K
        temp=[0 0;0 0];
        for i=1:N
            temp=temp+r(i,j)*(y(i,:)-y_bar(j,:))'*(y(i,:)-y_bar(j,:));
        end
        Sk(:,:,j)=temp; %(11.47)  from Kevin Murphy
        Stemp(:,:,j)=(sum(r(:,j))*beta_0(j)/(sum(r(:,j))+beta_0(j)))*((y_bar(j,:)-m0(j,:))'*(y_bar(j,:)-m0(j,:)));
        sigma_hat(:,:,j)=(S0(:,:,j)+Sk(:,:,j)+Stemp(:,:,j))/(v0+sum(r(:,j))+D+2); %(11.46) from Kevin Murphy
    end
    
    % estimate pi
    pi_hat=(sum(r)+alpha-1)/(N+sum(alpha)-K);  %(11.41) from kevin murphy
    
    %deal with extinguished clusters
    for j=1:K
        if sum(r(:,j))<0.01
            sigma_hat(:,:,j)=S0(:,:,j);
        end
    end
    
    % update mu, sigma, pi
    mu=mu_hat;
    PI=pi_hat;
    sigma=sigma_hat;
    
    keep_iter{iter}.mu=mu;
    keep_iter{iter}.PI=PI;
    keep_iter{iter}.sigma=sigma;
  
    %plot the new clusters after every iteration
    subplot(133)
    hold off
    plot(y(:,1),y(:,2),'b*')
    for i=1:K 
        hold on
        if PI(i)>0.01
            plot(mu(i,1),mu(i,2),'g*')
            eclipse_parameter(mu(i,:),sigma(:,:,i));
            text(mu(i,1), mu(i,2), num2str(i),'BackgroundColor', [.8 .8 .8]);
        end
    end

    title(strcat('MAP EM iteration#', num2str(iter)),'fontsize',15)
    drawnow;
    pause(0.1)
    
    for i=1:N
        for j=1:K
            %probabilty of Gaussian
            p_y(i,j)=(1/((2*pi)*sqrt(det(sigma(:,:,j)))))*exp(-0.5*(y(i,:)-mu(j,:))*inv(sigma(:,:,j))*(y(i,:)-mu(j,:))');
        end
    end
    
    Q=sum(log(p_y*PI'));
    
    diff_Q=abs(Q0-Q);
    
    Q0=Q;

end


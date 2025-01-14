function [coeff, score, latent] = pca_analysis(data)
% PCA_ANALYSIS Perform Principal Component Analysis (PCA) on the input dataset.
%   [coeff, score, latent] = PCA_ANALYSIS(data) computes the principal components
%   of the input dataset. The function returns:
%     - coeff: Principal component coefficients (eigenvectors).
%     - score: Representation of the data in the new coordinate system (principal components).
%     - latent: Eigenvalues of the covariance matrix, representing the variance explained by each principal component.

% Input:
%   data - A matrix where rows represent observations and columns represent variables.
%          Ensure that the data is numeric and does not contain missing values.

% Output:
%   coeff - Eigenvectors of the covariance matrix, representing directions of maximum variance.
%   score - Transformed data in terms of principal components.
%   latent - Eigenvalues indicating the variance explained by each principal component.

% Author: [Your Name]
% Date: [Date]

% Ensure the input is mean-centered
mean_data = mean(data);
data_centered = data - mean_data;

% Compute the covariance matrix
cov_matrix = cov(data_centered);

% Perform eigen decomposition
[eigenvectors, eigenvalues] = eig(cov_matrix);

% Extract eigenvalues as a vector and sort them in descending order
[latent, idx] = sort(diag(eigenvalues), 'descend');

% Reorder eigenvectors accordingly
coeff = eigenvectors(:, idx);

% Project data onto the principal components
score = data_centered * coeff;

% Display explained variance ratio
explained_variance = (latent / sum(latent)) * 100;

fprintf('Explained variance by each principal component:\n');
for i = 1:length(latent)
    fprintf('PC%d: %.2f%%\n', i, explained_variance(i));
end

% Plot cumulative explained variance
figure;
plot(cumsum(explained_variance), '-o');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Explained Variance by Principal Components');
grid on;

end

% Example Usage:
% Assuming `data` is your dataset:
% >> [coeff, score, latent] = pca_analysis(data);
% This function can be used to understand the underlying structure of high-dimensional
% datasets, reduce dimensionality, or visualize data in terms of principal components.

% Note:
% Principal Component Analysis (PCA) is a widely-used statistical technique
% for dimensionality reduction and feature extraction. It transforms the data
% into a new coordinate system such that the greatest variance by any projection
% of the data comes to lie on the first principal component, the second greatest
% variance on the second principal component, and so on.

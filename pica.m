clc; clear; close all;
% A.Babaei

%% Step 1: Load and Explore the Dataset
% Load data
data = load('file.mat');

% Access training data
training_images = data.training.images; % Size: 28x28x60000
training_labels = data.training.labels; % Size: 60000x1

% Choose 5000 samples randomly
n_samples = 5000;
indices = randperm(size(training_images, 3), n_samples); % Random indices
selected_images = training_images(:, :, indices);
selected_labels = training_labels(indices);

% Display a few sample images
figure('Name', 'Sample MNIST Images', 'NumberTitle', 'off');
for i = 1:9
    subplot(3, 3, i);
    imshow(reshape(selected_images(:, :, i), [28, 28]), []);
    title(['Label: ', num2str(selected_labels(i))], 'FontSize', 14);
end

%% Step 2: Preprocess the Data
% Reshape the images into vectors (5000 x 784)
img_vectors = reshape(selected_images, [28*28, n_samples])'; % Each row is a vectorized image

% Standardize the data using zscore
img_vectors = zscore(img_vectors);

%% Step 3: Apply PCA
% Perform PCA
[coeff, score, ~, ~, explained] = pca(img_vectors);

% Compute cumulative variance
cumulative_variance = cumsum(explained);

%% Step 4: Visualize Principal Components
% Plot the variance explained by each principal component
figure('Name', 'Variance Explained by Each PC', 'NumberTitle', 'off');
plot(explained, '-o', 'LineWidth', 1.5);
title('Variance Explained by Principal Components', 'FontSize', 18);
xlabel('Principal Component', 'FontSize', 16);
ylabel('Variance Explained (%)', 'FontSize', 16);
grid on; box off; set(gca, 'FontSize', 16);

% Plot the cumulative variance
figure('Name', 'Cumulative Variance Explained', 'NumberTitle', 'off');
plot(cumulative_variance, '-o', 'LineWidth', 1.5);
title('Cumulative Variance Explained by Principal Components', 'FontSize', 18);
xlabel('Number of Principal Components', 'FontSize', 16);
ylabel('Cumulative Variance Explained (%)', 'FontSize', 16);
grid on; set(gca, 'FontSize', 16); box off;

% Visualize the first 9 principal components as images
figure('Name', 'First 9 Principal Components', 'NumberTitle', 'off');
for i = 1:9
    subplot(3, 3, i);
    imshow(reshape(coeff(:, i), [28, 28]), []);
    title(['Principal Component ', num2str(i)], 'FontSize', 14);
    set(gca, 'FontSize', 16); box off;
end

%% Step 5: Reconstruct Images
% Reconstruct images using different numbers of components
n_components = [10, 50, 100, 300, 500];
mean_img = mean(img_vectors, 1); % Mean image

figure('Name', 'Image Reconstruction', 'NumberTitle', 'off');
for i = 1:length(n_components)
    k = n_components(i);
    reconstructed_data = score(:, 1:k) * coeff(:, 1:k)' + mean_img; % Reconstruction
    subplot(2, ceil(length(n_components)/2), i);
    imshow(reshape(reconstructed_data(1, :), [28, 28]), []);
    title([num2str(k), ' Components'], 'FontSize', 14);
    set(gca, 'FontSize', 16); box off;
end

%% Step 6: Analysis and Interpretation
% Print variance explained by selected components
fprintf('Explained variance by first 10 PCs: %.2f%%\n', sum(explained(1:10)));
fprintf('Explained variance by first 50 PCs: %.2f%%\n', sum(explained(1:50)));
fprintf('Explained variance by first 100 PCs: %.2f%%\n', sum(explained(1:100)));
fprintf('Explained variance by first 300 PCs: %.2f%%\n', sum(explained(1:300)));
fprintf('Explained variance by first 500 PCs: %.2f%%\n', sum(explained(1:500)));

clc; 
clf;
close all;
% clear all;
%% loading data

% total user = 21K
% user rating movie = 17K
% movie = 16K
% (user,movie) = 72K
% (user,movie_review) = 1.6M
% 
% review(total) = 72K
% review(rated at least 1) = 71K
% review(rated at least 10) = 48K
% review(rated at least 30) = 20K
% review(rated at least 100) = 0.5K

movie_rating = csvread('./CiaoDVD/clean_movie_ratings.txt');
movie_rating = [movie_rating(:,1:2) movie_rating(:,4:5)];

review_rating = csvread('./CiaoDVD/review_ratings.txt');

review_unique = unique(review_rating(:,2));
out = [review_unique, histc(review_rating(:,2),review_unique)];
review_rating_count_degree = sort(out(:,2),'descend');

% Dataset Name: CiaoDVDs
% 
% 1. Ciao movie ratings format:  
%     1) File: movie_ratings.txt (size: 72,665 --> 72.7K)
%     2) Columns: userID, movieID, genreID, reviewID, movieRating, date
% 
% 2. Ciao review ratings format: 
%     1) File: review_ratings.txt (size: 1,625,480 --> 1.6M)
%     2) Columns: userID, reviewID, reviewRating
%     3) Note: There are users who do not provide movie ratings but provide review ratings.
%     
% 3. Ciao user trusts fromat:
%     1) File: trusts.txt (size: 40,133 --> 40K)
%     2) Columns: trustorID, trusteeID, trustValue
%     3) Note: There are users who may not provide neither movie rating nor review ratings. 
% 
% 4. We will use/release the dataset with anonymous ids. 